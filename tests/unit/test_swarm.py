from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from benchmarking.swarm import Swarm


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
        self.main = Mock()
        self.cleanup = Mock()
        DummyAgent.instances.append(self)


class FakeArcade:
    def __init__(self) -> None:
        self.operation_mode = SimpleNamespace(ONLINE="online").ONLINE
        self.opened_tags: list[str] | None = None
        self.closed_card_id: str | None = None

    def make(self, game_id: str, scorecard_id: str) -> str:
        return f"env:{game_id}:{scorecard_id}"

    def open_scorecard(self, tags: list[str]) -> str:
        self.opened_tags = tags
        return "card-123"

    def close_scorecard(self, card_id: str) -> Mock:
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
