from __future__ import annotations

import json
import logging
import os
import time
from threading import Lock, Thread
from typing import TYPE_CHECKING, Optional, Type

import requests
from arc_agi import Arcade, OperationMode, RemoteEnvironmentWrapper
from arc_agi.scorecard import EnvironmentScorecard
from requests import HTTPError

from .agent import BenchmarkingAgent
from .base import ExitReason
from .rehydration import PreparedRehydration

if TYPE_CHECKING:
    from .base import Agent

logger = logging.getLogger()
DEFAULT_AGENT_NAME = BenchmarkingAgent.__name__.lower()

# Stop retrying a failed scorecard close once this much time has passed.
CLOSE_BUDGET_SECONDS = 15.0


class Swarm:
    """Orchestration for many agents playing many ARC-AGI-3 games."""

    GAMES: list[str]
    ROOT_URL: str
    COUNT: int
    agent_name: str
    agent_class: Type[Agent]
    threads: list[Thread]
    agents: list[Agent]
    record_games: list[str]
    cleanup_threads: list[Thread]
    headers: dict[str, str]
    card_id: Optional[str]
    _arc: Arcade

    def __init__(
        self,
        ROOT_URL: str,
        games: list[str],
        tags: Optional[list[str]] = None,
        config: Optional[str] = None,
        rehydration: Optional[PreparedRehydration] = None,
    ) -> None:
        self.GAMES = games
        self.rehydration = rehydration
        self.ROOT_URL = ROOT_URL
        self.agent_name = DEFAULT_AGENT_NAME
        self.agent_class = BenchmarkingAgent
        self.threads = []
        self.agents = []
        self.cleanup_threads = []
        self.headers = {
            "X-API-Key": os.getenv("ARC_API_KEY", ""),
            "Accept": "application/json",
        }
        self.tags = tags.copy() if tags is not None else []
        self.config = config
        self._arc = Arcade(operation_mode=OperationMode.ONLINE)
        self.tags.extend(["agent", self.agent_name])

        self.card_id = None
        self._shutdown_requested = False
        self._closed = False
        self._close_result: Optional[EnvironmentScorecard] = None
        # Held for the whole open and the whole close, so a shutdown on another
        # thread waits for either to finish instead of racing it.
        self._card_lock = Lock()

    def main(self) -> EnvironmentScorecard | None:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        print("***** MAKING SCORECARD")
        card_id = self.open_scorecard()
        if card_id is None or self._shutdown_requested:
            return None  # shutting down; the signal handler closes the card

        print(f"***** MAKING ALL AGENTS with card id: {card_id}")
        # create all the agents
        extra_kwargs = {"rehydration": self.rehydration} if self.rehydration else {}
        for i in range(len(self.GAMES)):
            g = self.GAMES[i % len(self.GAMES)]
            a = self.agent_class(
                card_id=card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                arc_env=self._arc.make(g, scorecard_id=card_id),
                config=self.config,
                **extra_kwargs,
            )
            self.agents.append(a)

        # create all the threads
        for a in self.agents:
            self.threads.append(Thread(target=a.main, daemon=True))

        # start all the threads
        for t in self.threads:
            t.start()

        # wait for all agent to finish
        for t in self.threads:
            t.join()

        # all agents are now done
        scorecard = self.close_scorecard()

        # Log agent exit reasons
        for a in self.agents:
            logger.info(f"AGENT EXIT REASON -- Agent: [{a.agent_name}] Game: [{a.game_id}] Reason: [{a.exit_reason}]")

        if scorecard:
            logger.info("--- FINAL SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))

        # Provide web link to scorecard
        if card_id:
            if self._arc.operation_mode == OperationMode.ONLINE:
                scorecard_url = f"{self.ROOT_URL}/scorecards/{card_id}"
                logger.info(f"View your scorecard online: {scorecard_url}")
            else:
                logger.info(
                    "Online scorecard is not available, to use the online API set the ONLINE_ONLY envvar to True"
                )

        self.cleanup(scorecard)

        return scorecard

    def open_scorecard(self) -> Optional[str]:
        with self._card_lock:
            if not self._shutdown_requested:
                self.card_id = self._arc.open_scorecard(tags=self.tags)
            return self.card_id

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _scorecard_exists(self, card_id: str) -> bool:
        try:
            with requests.Session() as session:
                session.headers.update(self.headers)
                response = session.get(f"{self.ROOT_URL}/api/v3/scorecards/{card_id}", timeout=10)
                return 199 < response.status_code < 300
        except requests.RequestException:
            logger.exception("HTTPError encountered on check for closed scorecard.")

        return False

    def close_scorecard(self) -> Optional[EnvironmentScorecard]:
        """Close the scorecard once; later callers get the first close's result."""
        with self._card_lock:
            if not self._closed:
                self._closed = True
                self._close_result = self._close(self.card_id)
            return self._close_result

    def _close(self, card_id: Optional[str]) -> Optional[EnvironmentScorecard]:
        if card_id is None:
            logger.info("SHUTDOWN: no scorecard opened")
            return None

        # Arcade.close_scorecard copies its master cookie jar into the session first,
        # so refresh that jar with the agent's current cookies.
        if self.agents and isinstance(self.agents[-1].arc_env, RemoteEnvironmentWrapper):
            with self._arc._cookie_lock:
                self._arc._master_cookie_jar.update(self.agents[-1].arc_env._master_cookie_jar)

        deadline = time.monotonic() + CLOSE_BUDGET_SECONDS
        for delay in (0, 1, 2):  # first attempt, then two retries
            if time.monotonic() + delay > deadline:
                break
            time.sleep(delay)
            try:
                scorecard = self._arc.close_scorecard(card_id)
                logger.info(f"SHUTDOWN: closed scorecard {card_id}")
                return scorecard
            except HTTPError as ex:
                # Check if scorecard closed due to idle/total time limit
                if ex.response is not None and ex.response.status_code == 404 and self._scorecard_exists(card_id):
                    for agent in self.agents:
                        if agent.exit_reason == ExitReason.API_ERROR:
                            agent.exit_reason = ExitReason.SCORECARD_CLOSED
                    logger.info(f"SHUTDOWN: closed scorecard {card_id} (closed by server)")
                    return None
                error: Exception = ex
            except Exception as ex:
                error = ex

        logger.error("Exception encountered on scorecard close. Swarm exit reason API_ERROR.", exc_info=error)
        for agent in self.agents:
            agent.exit_reason = ExitReason.API_ERROR
        logger.error(f"SHUTDOWN: close failed for {card_id}: {type(error).__name__}: {error}")
        return None

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Cleanup all agents."""
        for a in self.agents:
            a.cleanup(scorecard)
        if hasattr(self, "_session"):
            self._session.close()
