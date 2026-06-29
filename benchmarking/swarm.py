from __future__ import annotations

import json
import logging
import os
from threading import Thread
from typing import TYPE_CHECKING, Optional, Type

import requests
from arc_agi import Arcade, OperationMode, RemoteEnvironmentWrapper
from arc_agi.scorecard import EnvironmentScorecard
from requests import HTTPError

from .agent import BenchmarkingAgent
from .base import ExitReason

if TYPE_CHECKING:
    from .base import Agent

logger = logging.getLogger()
DEFAULT_AGENT_NAME = BenchmarkingAgent.__name__.lower()


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
    ) -> None:
        self.GAMES = games
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

    def main(self) -> EnvironmentScorecard | None:
        """The main orchestration loop, continues until all agents are done."""

        # submit start of scorecard
        print("***** MAKING SCORECARD")
        self.card_id = self.open_scorecard()

        print(f"***** MAKING ALL AGENTS with card id: {self.card_id}")
        # create all the agents
        for i in range(len(self.GAMES)):
            g = self.GAMES[i % len(self.GAMES)]
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                agent_name=self.agent_name,
                ROOT_URL=self.ROOT_URL,
                record=True,
                arc_env=self._arc.make(g, scorecard_id=self.card_id),
                config=self.config,
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

        # Refresh arcade cookies with the last agent's cookies
        if self.agents:
            cookie_agent: Agent = self.agents[-1]
            if isinstance(cookie_agent.arc_env, RemoteEnvironmentWrapper):
                self._arc._master_cookie_jar.update(cookie_agent.arc_env._master_cookie_jar)

        # all agents are now done
        card_id = self.card_id
        scorecard = self.close_scorecard(card_id)

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

    def open_scorecard(self) -> str:
        return self._arc.open_scorecard(tags=self.tags)  # type: ignore[no-any-return]

    def _scorecard_exists(self, card_id: str) -> bool:
        try:
            with requests.Session() as session:
                session.headers.update(self.headers)
                response = session.get(f"{self.ROOT_URL}/api/v3/scorecards/{card_id}", timeout=10)
                return 199 < response.status_code < 300
        except requests.RequestException:
            logger.exception("HTTPError encountered on check for closed scorecard.")

        return False

    def close_scorecard(self, card_id: str) -> Optional[EnvironmentScorecard]:
        self.card_id = None

        # Close scorecard gracefully, determine if scorecard automatically closed on initial failure
        _scorecard: Optional[EnvironmentScorecard] = None
        try:
            _scorecard = self._arc.close_scorecard(card_id)
        except HTTPError as ex:

            # Check if scorecard closed due to idle/total time limit
            if ex.response is not None and ex.response.status_code == 404 and self._scorecard_exists(card_id):
                for agent in self.agents:
                    if agent.exit_reason == ExitReason.API_ERROR:
                        agent.exit_reason = ExitReason.SCORECARD_CLOSED
            else:
                logger.error("Exception encountered on scorecard close. Swarm exit reason API_ERROR.")
                raise ex

        return _scorecard

    def cleanup(self, scorecard: Optional[EnvironmentScorecard] = None) -> None:
        """Cleanup all agents."""
        for a in self.agents:
            a.cleanup(scorecard)
        if hasattr(self, "_session"):
            self._session.close()
