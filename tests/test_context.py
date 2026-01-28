"""Tests for SessionContext."""
import json
from dataclasses import FrozenInstanceError
from typing import List

from threadsafe_datastore import Datastore

from arcagi3.checkpoint import CheckpointManager
from arcagi3.utils.context import FrameState, GameProgress, SessionContext


def create_64x64_grid(value: int = 0) -> List[List[int]]:
    """Create a 64x64 grid filled with a value."""
    return [[value for _ in range(64)] for _ in range(64)]


def test_context_initialization_with_defaults():
    context = SessionContext()

    assert context.datastore is not None
    assert isinstance(context.datastore, Datastore)
    assert context.frames.frame_grids == ()
    assert context.frames.previous_grids == ()
    assert context.game.current_score == 0
    assert context.game.current_state == "IN_PROGRESS"
    assert context.game.game_id == ""
    assert context.game.play_num == 1
    assert context.game.play_action_counter == 0
    assert context.game.action_counter == 0
    assert context.game.guid is None
    assert context.game.previous_score == 0
    assert context.game.available_actions == ()
    assert context.frame_images == []


def test_context_initialization_with_custom_datastore():
    datastore = Datastore()
    datastore["test_key"] = "test_value"

    context = SessionContext(datastore=datastore)

    assert context.datastore is datastore
    assert context.datastore["test_key"] == "test_value"


def test_context_initialization_with_parameters():
    datastore = Datastore()
    frame_grids = (create_64x64_grid(1), create_64x64_grid(2))

    context = SessionContext(
        datastore=datastore,
        frames=FrameState(frame_grids=frame_grids, previous_grids=(create_64x64_grid(0),)),
        game=GameProgress(
            game_id="test_game",
            current_score=100,
            current_state="WIN",
            play_num=2,
            play_action_counter=5,
            action_counter=10,
            guid="test-guid",
            previous_score=90,
            available_actions=("ACTION1", "ACTION2"),
        ),
    )

    assert len(context.frame_images) == 2
    assert len(context.frames.frame_grids) == 2
    assert context.game.current_score == 100
    assert context.game.current_state == "WIN"
    assert context.game.game_id == "test_game"
    assert context.game.play_num == 2
    assert context.game.play_action_counter == 5
    assert context.game.action_counter == 10
    assert context.game.guid == "test-guid"
    assert context.game.previous_score == 90
    assert len(context.previous_images) == 1
    assert context.game.available_actions == ("ACTION1", "ACTION2")


def test_context_update():
    context = SessionContext()

    initial_grids = [create_64x64_grid(1)]
    context.update(
        frame_grids=initial_grids,
        current_score=50,
        current_state="IN_PROGRESS",
        guid="initial-guid",
    )

    assert len(context.frames.frame_grids) == 1
    assert len(context.frame_images) == 1
    assert context.game.current_score == 50
    assert context.game.current_state == "IN_PROGRESS"
    assert context.game.guid == "initial-guid"
    assert context.game.previous_score == 0

    new_grids = [create_64x64_grid(2), create_64x64_grid(3)]
    context.update(
        frame_grids=new_grids,
        current_score=100,
        current_state="WIN",
        guid="new-guid",
    )

    assert len(context.frames.frame_grids) == 2
    assert len(context.frame_images) == 2
    assert context.game.current_score == 100
    assert context.game.current_state == "WIN"
    assert context.game.guid == "new-guid"
    assert context.game.previous_score == 50
    assert len(context.previous_images) == 1


def test_context_update_preserves_previous_state():
    context = SessionContext()

    context.update(
        frame_grids=[create_64x64_grid(1)],
        current_score=10,
        current_state="IN_PROGRESS",
    )

    prev_images = context.frame_images.copy()
    prev_grids = context.frames.frame_grids
    prev_score = context.game.current_score

    context.update(
        frame_grids=[create_64x64_grid(2)],
        current_score=20,
        current_state="IN_PROGRESS",
    )

    assert context.game.previous_score == prev_score
    assert len(context.previous_images) == len(prev_images)
    assert len(context.frames.previous_grids) == len(prev_grids)


def test_context_properties():
    context = SessionContext()

    context.update(frame_grids=[], current_score=0, current_state="WIN")
    assert context.is_won is True
    assert context.is_game_over is True

    context.update(frame_grids=[], current_score=0, current_state="GAME_OVER")
    assert context.is_won is False
    assert context.is_game_over is True

    context.update(frame_grids=[], current_score=0, current_state="IN_PROGRESS")
    assert context.is_won is False
    assert context.is_game_over is False

    context.update(frame_grids=[], current_score=100, current_state="IN_PROGRESS")
    assert context.score_increased is True
    context.update(frame_grids=[], current_score=100, current_state="IN_PROGRESS")
    assert context.score_increased is False
    context.update(frame_grids=[], current_score=90, current_state="IN_PROGRESS")
    assert context.score_increased is False


def test_context_last_frame():
    context = SessionContext()

    assert context.last_frame_image() is None
    assert context.last_frame_grid is None

    grids = [create_64x64_grid(1), create_64x64_grid(2)]
    context.update(
        frame_grids=grids,
        current_score=0,
        current_state="IN_PROGRESS",
    )

    assert context.last_frame_image() is not None
    assert context.last_frame_grid is not None
    assert context.last_frame_grid == grids[-1]


def test_context_get_frame_images_resize():
    context = SessionContext()

    grids = [create_64x64_grid(1), create_64x64_grid(2)]
    context.update(
        frame_grids=grids,
        current_score=0,
        current_state="IN_PROGRESS",
    )

    original = context.get_frame_images()
    assert len(original) == 2
    original_size = original[0].size
    assert original_size[0] > 0 and original_size[1] > 0

    resized = context.get_frame_images(resize=32)
    assert len(resized) == 2
    assert resized[0].size == (32, 32)

    resized_tuple = context.get_frame_images(resize=(16, 24))
    assert len(resized_tuple) == 2
    assert resized_tuple[0].size == (16, 24)

    assert context.frame_images[0].size == original_size


def test_context_datastore_roundtrip_checkpoint(tmp_path):
    checkpoint_id = "ds-roundtrip"
    checkpoint_dir = str(tmp_path)

    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    ctx.datastore["k1"] = "v1"
    ctx.datastore["k2"] = 2
    ctx.datastore["nested"] = {"a": [1, 2, 3], "b": {"c": True}}

    mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
    state = ctx.get_state()
    state["metadata"] = {
        "config": "dummy",
        "checkpoint_id": checkpoint_id,
        "max_actions": 0,
        "num_plays": 1,
        "max_episode_actions": 0,
    }
    mgr.save_state(state)

    restored = SessionContext.restore_from_checkpoint(
        checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir
    )

    assert dict(restored.datastore.items()) == dict(ctx.datastore.items())


def test_context_checkpoint_rejects_non_json_datastore(tmp_path):
    checkpoint_id = "ds-non-json"
    checkpoint_dir = str(tmp_path)

    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    ctx.datastore["bad"] = object()

    try:
        ctx.get_state()
        assert False, "Expected get_state() to reject non-JSON-serializable values"
    except TypeError:
        pass


def test_context_last_frame_image_resize():
    context = SessionContext()

    grids = [create_64x64_grid(1)]
    context.update(
        frame_grids=grids,
        current_score=0,
        current_state="IN_PROGRESS",
    )

    original = context.last_frame_image()
    assert original is not None
    original_size = original.size
    assert original_size[0] > 0 and original_size[1] > 0

    resized = context.last_frame_image(resize=32)
    assert resized is not None
    assert resized.size == (32, 32)

    resized_tuple = context.last_frame_image(resize=(16, 24))
    assert resized_tuple is not None
    assert resized_tuple.size == (16, 24)

    assert context.last_frame_image().size == original_size


def test_context_datastore_access():
    context = SessionContext()

    context.datastore["key1"] = "value1"
    context.datastore["key2"] = 42

    assert context.datastore["key1"] == "value1"
    assert context.datastore["key2"] == 42

    context.datastore["counter"] = 0
    context.datastore.increment("counter", 5)
    assert context.datastore["counter"] == 5


def test_context_update_with_empty_frames():
    context = SessionContext()

    context.update(
        frame_grids=[],
        current_score=0,
        current_state="IN_PROGRESS",
    )

    assert len(context.frames.frame_grids) == 0
    assert len(context.frame_images) == 0
    assert context.last_frame_image() is None
    assert context.last_frame_grid is None
    assert context.previous_images == []


def test_context_immutability():
    context = SessionContext()
    with_context = context.game

    try:
        with_context.current_score = 5  # type: ignore[attr-defined]
        assert False, "Expected GameProgress to be frozen"
    except FrozenInstanceError:
        pass

    context.set_available_actions(["ACTION1"])
    try:
        context.game.available_actions += ("ACTION2",)
        assert False, "Expected available_actions to be immutable"
    except (TypeError, FrozenInstanceError):
        pass


def test_context_checkpoint_json_serializable():
    context = SessionContext()
    context.set_available_actions(["ACTION1"])
    state = context.get_state()
    state["metadata"] = {
        "config": "dummy",
        "checkpoint_id": "id",
        "max_actions": 0,
        "num_plays": 1,
        "max_episode_actions": 0,
    }
    json.dumps(state)


def test_context_setters_update_state():
    context = SessionContext()
    context.set_game_identity(game_id="game-x", guid="guid-x")
    context.set_play_num(3)
    context.set_counters(play_action_counter=4, action_counter=9)
    context.set_available_actions(["ACTION1", "ACTION2"])

    assert context.game.game_id == "game-x"
    assert context.game.guid == "guid-x"
    assert context.game.play_num == 3
    assert context.game.play_action_counter == 4
    assert context.game.action_counter == 9
    assert context.game.available_actions == ("ACTION1", "ACTION2")


def test_context_frame_state_immutability():
    context = SessionContext()
    context.update(frame_grids=[create_64x64_grid(1)], current_score=0, current_state="IN_PROGRESS")
    frames = context.frames
    try:
        frames.frame_grids += (create_64x64_grid(2),)
        assert False, "Expected FrameState to be frozen"
    except (TypeError, FrozenInstanceError):
        pass


def test_checkpoint_rejects_non_dict_metrics(tmp_path):
    checkpoint_id = "bad-metrics"
    checkpoint_dir = str(tmp_path)
    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    state = ctx.get_state()
    state["metadata"] = {
        "config": "dummy",
        "checkpoint_id": checkpoint_id,
        "max_actions": 0,
        "num_plays": 1,
        "max_episode_actions": 0,
    }
    state["metrics"]["total_cost"] = "not-a-dict"
    mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
    try:
        mgr.save_state(state)
        assert False, "Expected save_state() to reject non-dict metrics"
    except TypeError:
        pass


def test_checkpoint_rejects_non_list_history(tmp_path):
    checkpoint_id = "bad-history"
    checkpoint_dir = str(tmp_path)
    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    state = ctx.get_state()
    state["metadata"] = {
        "config": "dummy",
        "checkpoint_id": checkpoint_id,
        "max_actions": 0,
        "num_plays": 1,
        "max_episode_actions": 0,
    }
    state["history"]["action_history"] = "not-a-list"
    mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
    try:
        mgr.save_state(state)
        assert False, "Expected save_state() to reject non-list action_history"
    except TypeError:
        pass
