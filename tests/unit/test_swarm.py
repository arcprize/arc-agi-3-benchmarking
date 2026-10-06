import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests
from requests import HTTPError
from requests.cookies import RequestsCookieJar

from benchmarking import swarm as swarm_module
from benchmarking.base import ExitReason
from benchmarking.swarm import Swarm


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(
        swarm_module, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None)
    )


class DummyAgent:
    instances: list["DummyAgent"] = []

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: str,
        config: str | None = None,
    ) -> None:
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.ROOT_URL = ROOT_URL
        self.record = record
        self.arc_env = arc_env
        self.config = config
        self.exit_reason = ExitReason.UNKNOWN
        self.main = Mock()
        self.cleanup = Mock()
        DummyAgent.instances.append(self)


class FakeArcade:
    def __init__(self, operation_mode=None) -> None:  # noqa: ANN001
        self.requested_operation_mode = operation_mode
        self.operation_mode = SimpleNamespace(ONLINE="online").ONLINE
        self.opened_tags: list[str] | None = None
        self.closed_card_id: str | None = None
        self.open_calls = 0
        self.close_calls: list[str] = []
        self.close_errors: list[Exception] = []
        self.jar_at_close: list[dict[str, str]] = []
        self.open_gate: threading.Event | None = None
        self.close_gate: threading.Event | None = None
        self.opening = threading.Event()
        self.closing = threading.Event()
        self._cookie_lock = threading.Lock()
        self._master_cookie_jar = RequestsCookieJar()

    def make(self, game_id: str, scorecard_id: str) -> str:
        return f"env:{game_id}:{scorecard_id}"

    def open_scorecard(self, tags: list[str]) -> str:
        self.open_calls += 1
        self.opening.set()
        if self.open_gate is not None:
            self.open_gate.wait(5)
        self.opened_tags = tags
        return "card-123"

    def close_scorecard(self, card_id: str) -> Mock:
        self.close_calls.append(card_id)
        self.jar_at_close.append(self._master_cookie_jar.get_dict())
        self.closing.set()
        if self.close_gate is not None:
            self.close_gate.wait(5)
        if self.close_errors:
            raise self.close_errors.pop(0)
        self.closed_card_id = card_id
        scorecard = Mock()
        scorecard.model_dump.return_value = {"card_id": card_id}
        return scorecard


class FakeThread:
    instances: list["FakeThread"] = []

    def __init__(self, target, daemon: bool) -> None:  # noqa: ANN001
        self.target = target
        self.daemon = daemon
        self.started = False
        self.joined = False
        FakeThread.instances.append(self)

    def start(self) -> None:
        self.started = True
        self.target()

    def join(self) -> None:
        self.joined = True


@pytest.mark.unit
class TestSwarm:
    def setup_method(self) -> None:
        DummyAgent.instances.clear()
        FakeThread.instances.clear()

    def test_swarm_init_registers_agent_and_tags(self):
        with (
            patch("benchmarking.swarm.BenchmarkingAgent", DummyAgent),
            patch("benchmarking.swarm.Arcade", FakeArcade),
        ):
            swarm = Swarm(
                ROOT_URL="https://example.com",
                games=["game1", "game2"],
                tags=["experiment"],
            )

        assert swarm.agent_name == "benchmarkingagent"
        assert swarm.agent_class is DummyAgent
        assert swarm.GAMES == ["game1", "game2"]
        assert swarm.tags == ["experiment", "agent", "benchmarkingagent"]
        assert swarm.headers["Accept"] == "application/json"

    def test_main_creates_agents_threads_and_closes_scorecard(self):
        with (
            patch("benchmarking.swarm.BenchmarkingAgent", DummyAgent),
            patch("benchmarking.swarm.Arcade", FakeArcade),
            patch("benchmarking.swarm.Thread", FakeThread),
        ):
            swarm = Swarm(
                ROOT_URL="https://example.com",
                games=["game1", "game2", "game3"],
                config="openai-gpt-5.4-openrouter",
            )

            scorecard = swarm.main()

        assert scorecard is not None
        assert len(DummyAgent.instances) == 3
        assert [agent.game_id for agent in DummyAgent.instances] == [
            "game1",
            "game2",
            "game3",
        ]
        assert all(agent.card_id == "card-123" for agent in DummyAgent.instances)
        assert all(agent.record is True for agent in DummyAgent.instances)
        assert all(
            agent.config == "openai-gpt-5.4-openrouter"
            for agent in DummyAgent.instances
        )
        assert all(agent.main.call_count == 1 for agent in DummyAgent.instances)
        assert all(thread.started for thread in FakeThread.instances)
        assert all(thread.joined for thread in FakeThread.instances)
        assert all(agent.cleanup.call_count == 1 for agent in DummyAgent.instances)


def _http_error(status_code: int) -> HTTPError:
    response = SimpleNamespace(status_code=status_code)
    return HTTPError(response=response)


class _ExitAgent:
    """Minimal agent stub exposing only what close_scorecard reads."""

    def __init__(self, exit_reason: ExitReason) -> None:
        self.exit_reason = exit_reason
        self.agent_name = "agent"
        self.game_id = "game"
        self.arc_env: object = None


def _make_swarm(games: list[str] | None = None) -> Swarm:
    with (
        patch("benchmarking.swarm.BenchmarkingAgent", DummyAgent),
        patch("benchmarking.swarm.Arcade", FakeArcade),
    ):
        return Swarm(ROOT_URL="https://example.com", games=games or ["game1"])


def _outcome_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("SHUTDOWN:")]


def _swarm_with_agents(agents: list[_ExitAgent], close_error: HTTPError) -> Swarm:
    swarm = _make_swarm()
    swarm.agents = agents
    swarm.card_id = "card-123"
    swarm._arc.close_scorecard = Mock(side_effect=close_error)
    return swarm


@pytest.mark.unit
class TestSwarmCloseScorecard:
    @pytest.fixture(autouse=True)
    def _info_logs(self, caplog):
        caplog.set_level(logging.INFO)

    def test_idle_closed_scorecard_reclassifies_only_api_error_agents(self, caplog):
        api_agent = _ExitAgent(ExitReason.API_ERROR)
        win_agent = _ExitAgent(ExitReason.GAME_WIN)
        swarm = _swarm_with_agents([api_agent, win_agent], _http_error(404))

        with patch.object(Swarm, "_scorecard_exists", return_value=True):
            result = swarm.close_scorecard()

        assert result is None
        assert api_agent.exit_reason is ExitReason.SCORECARD_CLOSED
        assert win_agent.exit_reason is ExitReason.GAME_WIN
        assert _outcome_lines(caplog) == [
            "SHUTDOWN: closed scorecard card-123 (closed by server)"
        ]

    def test_404_but_scorecard_absent_reraises_and_keeps_reasons(self, caplog):
        api_agent = _ExitAgent(ExitReason.API_ERROR)
        swarm = _swarm_with_agents([api_agent], _http_error(404))

        with patch.object(Swarm, "_scorecard_exists", return_value=False):
            swarm.close_scorecard()

        assert api_agent.exit_reason is ExitReason.API_ERROR
        assert _outcome_lines(caplog) == ["SHUTDOWN: close failed for card-123: HTTPError: "]

    def test_non_404_error_reraises_without_existence_check(self, caplog):
        api_agent = _ExitAgent(ExitReason.API_ERROR)
        swarm = _swarm_with_agents([api_agent], _http_error(500))

        with patch.object(Swarm, "_scorecard_exists") as exists:
            swarm.close_scorecard()

        exists.assert_not_called()
        assert api_agent.exit_reason is ExitReason.API_ERROR
        assert len(_outcome_lines(caplog)) == 1


class FakeRemoteEnv:
    def __init__(self, jar: RequestsCookieJar) -> None:
        self._master_cookie_jar = jar


@pytest.mark.unit
class TestSwarmShutdown:
    @pytest.fixture(autouse=True)
    def _setup(self, caplog):
        caplog.set_level(logging.INFO)
        DummyAgent.instances.clear()
        FakeThread.instances.clear()

    def test_shutdown_before_open_never_opens(self, caplog):
        swarm = _make_swarm()
        swarm.request_shutdown()

        assert swarm.close_scorecard() is None
        with patch("benchmarking.swarm.Thread", FakeThread):
            assert swarm.main() is None

        assert swarm._arc.open_calls == 0
        assert swarm._arc.close_calls == []
        assert DummyAgent.instances == []
        assert _outcome_lines(caplog) == ["SHUTDOWN: no scorecard opened"]

    def test_shutdown_during_open_closes_once_opened(self, caplog):
        swarm = _make_swarm()
        arc = swarm._arc
        arc.open_gate = threading.Event()
        results: list[object] = []

        main_thread = threading.Thread(target=lambda: results.append(swarm.main()))
        main_thread.start()
        assert arc.opening.wait(5)

        swarm.request_shutdown()
        closer = threading.Thread(target=lambda: results.append(swarm.close_scorecard()))
        closer.start()
        arc.open_gate.set()
        main_thread.join(5)
        closer.join(5)

        assert arc.close_calls == ["card-123"]
        assert DummyAgent.instances == []
        assert _outcome_lines(caplog) == ["SHUTDOWN: closed scorecard card-123"]

    def test_shutdown_mid_run_shares_close_with_normal_path(self, caplog):
        swarm = _make_swarm(["game1", "game2"])
        shutdown_results: list[object] = []

        def shut_down() -> None:
            # FakeThread runs agents inline, so this is mid-run on the agent thread.
            if not shutdown_results:
                swarm.request_shutdown()
                shutdown_results.append(swarm.close_scorecard())

        class ShutdownAgent(DummyAgent):
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                super().__init__(*args, **kwargs)
                self.main = Mock(side_effect=shut_down)

        swarm.agent_class = ShutdownAgent
        with patch("benchmarking.swarm.Thread", FakeThread):
            final = swarm.main()

        assert swarm._arc.close_calls == ["card-123"]
        assert final is shutdown_results[0] is not None
        assert _outcome_lines(caplog) == ["SHUTDOWN: closed scorecard card-123"]

    def test_second_close_waits_for_close_in_progress(self, caplog):
        swarm = _make_swarm()
        arc = swarm._arc
        swarm.open_scorecard()
        arc.close_gate = threading.Event()
        results: list[object] = []

        first = threading.Thread(target=lambda: results.append(swarm.close_scorecard()))
        first.start()
        assert arc.closing.wait(5)
        second = threading.Thread(target=lambda: results.append(swarm.close_scorecard()))
        second.start()
        arc.close_gate.set()
        first.join(5)
        second.join(5)

        assert arc.close_calls == ["card-123"]
        assert len(results) == 2 and results[0] is results[1] is not None
        assert _outcome_lines(caplog) == ["SHUTDOWN: closed scorecard card-123"]

    def test_transient_close_error_is_retried(self, caplog):
        swarm = _make_swarm()
        swarm.open_scorecard()
        agent = _ExitAgent(ExitReason.GAME_WIN)
        swarm.agents = [agent]
        swarm._arc.close_errors = [requests.ConnectionError("reset")]

        assert swarm.close_scorecard() is not None

        assert swarm._arc.close_calls == ["card-123", "card-123"]
        assert agent.exit_reason is ExitReason.GAME_WIN
        assert _outcome_lines(caplog) == ["SHUTDOWN: closed scorecard card-123"]

    def test_close_gives_up_when_retries_exhausted(self, caplog):
        swarm = _make_swarm()
        swarm.open_scorecard()
        swarm._arc.close_errors = [requests.Timeout("slow")] * 10

        assert swarm.close_scorecard() is None

        assert len(swarm._arc.close_calls) == 3  # first attempt plus two retries
        assert _outcome_lines(caplog) == ["SHUTDOWN: close failed for card-123: Timeout: slow"]

    def test_close_uses_fresh_agent_cookies(self):
        swarm = _make_swarm()
        swarm.open_scorecard()
        swarm._arc._master_cookie_jar.set("sticky", "stale")
        env_jar = RequestsCookieJar()
        env_jar.set("sticky", "fresh")
        agent = _ExitAgent(ExitReason.GAME_WIN)
        agent.arc_env = FakeRemoteEnv(env_jar)
        swarm.agents = [agent]

        with patch("benchmarking.swarm.RemoteEnvironmentWrapper", FakeRemoteEnv):
            swarm.close_scorecard()

        assert swarm._arc.jar_at_close == [{"sticky": "fresh"}]


class DummyRehydratingAgent(DummyAgent):
    def __init__(self, *args, rehydration=None, **kwargs) -> None:  # noqa: ANN001, ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.rehydration = rehydration


@pytest.mark.unit
class TestSwarmRehydration:
    def setup_method(self) -> None:
        DummyAgent.instances.clear()
        FakeThread.instances.clear()

    def test_forwards_rehydration_to_agent(self):
        prepared = object()
        with (
            patch("benchmarking.swarm.BenchmarkingAgent", DummyRehydratingAgent),
            patch("benchmarking.swarm.Arcade", FakeArcade),
            patch("benchmarking.swarm.Thread", FakeThread),
        ):
            Swarm(
                ROOT_URL="https://example.com", games=["ls20-abc"], rehydration=prepared
            ).main()

        assert [agent.rehydration for agent in DummyAgent.instances] == [prepared]

    def test_omits_rehydration_kwarg_when_unset(self):
        # DummyAgent has no rehydration parameter; passing one would raise.
        with (
            patch("benchmarking.swarm.BenchmarkingAgent", DummyAgent),
            patch("benchmarking.swarm.Arcade", FakeArcade),
            patch("benchmarking.swarm.Thread", FakeThread),
        ):
            Swarm(ROOT_URL="https://example.com", games=["g"]).main()
        assert len(DummyAgent.instances) == 1
