"""Tests for SessionContext."""
from typing import List
from PIL import Image
from threadsafe_datastore import Datastore

from arcagi3.utils.context import SessionContext
from arcagi3.checkpoint import CheckpointManager


def create_64x64_grid(value: int = 0) -> List[List[int]]:
    """Create a 64x64 grid filled with a value."""
    return [[value for _ in range(64)] for _ in range(64)]


def test_context_initialization_with_defaults():
    """Test that context can be initialized with no arguments."""
    context = SessionContext()
    
    assert context.datastore is not None
    assert isinstance(context.datastore, Datastore)
    assert context.frame_images == []
    assert context.frame_grids == []
    assert context.current_score == 0
    assert context.current_state == "IN_PROGRESS"
    assert context.game_id == ""
    assert context.play_num == 1
    assert context.play_action_counter == 0
    assert context.action_counter == 0
    assert context.guid is None
    assert context.previous_score == 0
    assert context.previous_images == []
    assert context.previous_grids == []
    assert context.available_actions == []


def test_context_initialization_with_custom_datastore():
    """Test that context can be initialized with a custom datastore."""
    datastore = Datastore()
    datastore["test_key"] = "test_value"
    
    context = SessionContext(datastore=datastore)
    
    assert context.datastore is datastore
    assert context.datastore["test_key"] == "test_value"


def test_context_initialization_with_parameters():
    """Test that context can be initialized with all parameters."""
    datastore = Datastore()
    frame_grids = [create_64x64_grid(1), create_64x64_grid(2)]
    frame_images = [Image.new("RGB", (64, 64)) for _ in range(2)]
    
    context = SessionContext(
        datastore=datastore,
        frame_images=frame_images,
        frame_grids=frame_grids,
        current_score=100,
        current_state="WIN",
        game_id="test_game",
        play_num=2,
        play_action_counter=5,
        action_counter=10,
        guid="test-guid",
        previous_score=90,
        previous_images=[Image.new("RGB", (64, 64))],
        previous_grids=[create_64x64_grid(0)],
        available_actions=["ACTION1", "ACTION2"],
    )
    
    assert len(context.frame_images) == 2
    assert len(context.frame_grids) == 2
    assert context.current_score == 100
    assert context.current_state == "WIN"
    assert context.game_id == "test_game"
    assert context.play_num == 2
    assert context.play_action_counter == 5
    assert context.action_counter == 10
    assert context.guid == "test-guid"
    assert context.previous_score == 90
    assert len(context.previous_images) == 1
    assert len(context.previous_grids) == 1
    assert context.available_actions == ["ACTION1", "ACTION2"]


def test_context_update():
    """Test that context update works correctly."""
    context = SessionContext()
    
    # Set initial state
    initial_grids = [create_64x64_grid(1)]
    context.update(
        frame_grids=initial_grids,
        current_score=50,
        current_state="IN_PROGRESS",
        guid="initial-guid",
    )
    
    assert len(context.frame_grids) == 1
    assert len(context.frame_images) == 1
    assert context.current_score == 50
    assert context.current_state == "IN_PROGRESS"
    assert context.guid == "initial-guid"
    assert context.previous_score == 0  # Initial previous score
    
    # Update to new state
    new_grids = [create_64x64_grid(2), create_64x64_grid(3)]
    context.update(
        frame_grids=new_grids,
        current_score=100,
        current_state="WIN",
        guid="new-guid",
    )
    
    assert len(context.frame_grids) == 2
    assert len(context.frame_images) == 2
    assert context.current_score == 100
    assert context.current_state == "WIN"
    assert context.guid == "new-guid"
    # Previous state should be saved
    assert context.previous_score == 50
    assert len(context.previous_images) == 1
    assert len(context.previous_grids) == 1


def test_context_update_preserves_previous_state():
    """Test that update correctly preserves previous state."""
    context = SessionContext()
    
    # First update
    context.update(
        frame_grids=[create_64x64_grid(1)],
        current_score=10,
        current_state="IN_PROGRESS",
    )
    
    prev_images = context.frame_images.copy()
    prev_grids = context.frame_grids.copy()
    prev_score = context.current_score
    
    # Second update
    context.update(
        frame_grids=[create_64x64_grid(2)],
        current_score=20,
        current_state="IN_PROGRESS",
    )
    
    # Previous state should match what was current before update
    assert context.previous_score == prev_score
    assert len(context.previous_images) == len(prev_images)
    assert len(context.previous_grids) == len(prev_grids)


def test_context_properties():
    """Test convenience properties."""
    context = SessionContext()
    
    # Test is_won
    context._current_state = "WIN"
    assert context.is_won is True
    assert context.is_game_over is True
    
    context._current_state = "GAME_OVER"
    assert context.is_won is False
    assert context.is_game_over is True
    
    context._current_state = "IN_PROGRESS"
    assert context.is_won is False
    assert context.is_game_over is False
    
    # Test score_increased
    context._current_score = 100
    context._previous_score = 50
    assert context.score_increased is True
    
    context._previous_score = 100
    assert context.score_increased is False
    
    context._previous_score = 150
    assert context.score_increased is False


def test_context_last_frame():
    """Test last_frame_image and last_frame_grid properties."""
    context = SessionContext()
    
    # Empty context
    assert context.last_frame_image() is None
    assert context.last_frame_grid is None
    
    # With frames
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
    """Test get_frame_images with resize parameter."""
    context = SessionContext()
    
    grids = [create_64x64_grid(1), create_64x64_grid(2)]
    context.update(
        frame_grids=grids,
        current_score=0,
        current_state="IN_PROGRESS",
    )
    
    # Original size - check that images were created
    original = context.get_frame_images()
    assert len(original) == 2
    original_size = original[0].size
    assert original_size[0] > 0 and original_size[1] > 0
    
    # Resize with int (square)
    resized = context.get_frame_images(resize=32)
    assert len(resized) == 2
    assert resized[0].size == (32, 32)

    # Resize with tuple
    resized_tuple = context.get_frame_images(resize=(16, 24))
    assert len(resized_tuple) == 2
    assert resized_tuple[0].size == (16, 24)

    # Original should be unchanged
    assert context.frame_images[0].size == original_size


def test_context_datastore_roundtrip_checkpoint(tmp_path):
    checkpoint_id = "ds-roundtrip"
    checkpoint_dir = str(tmp_path)

    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    ctx.datastore["k1"] = "v1"
    ctx.datastore["k2"] = 2
    ctx.datastore["nested"] = {"a": [1, 2, 3], "b": {"c": True}}

    mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
    mgr.save_state(
        {
            "metadata": {
                "config": "dummy",
                "checkpoint_id": checkpoint_id,
                "game_id": "g",
                "guid": "guid",
                "frame_grids": [],
                "available_actions": [],
                "datastore": ctx.datastore_snapshot(),
                "max_actions": 0,
                "num_plays": 1,
                "max_episode_actions": 0,
                "action_counter": 0,
                "current_play": 1,
                "play_action_counter": 0,
                "current_score": 0,
                "current_state": "IN_PROGRESS",
                "previous_score": 0,
            },
            "metrics": {
                "total_cost": ctx.total_cost,
                "total_usage": ctx.total_usage,
                "action_history": ctx.action_history,
            },
        }
    )

    restored = SessionContext.restore_from_checkpoint(
        checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir
    )

    assert dict(restored.datastore.items()) == dict(ctx.datastore.items())


def test_context_datastore_checkpoint_rejects_non_json(tmp_path):
    checkpoint_id = "ds-non-json"
    checkpoint_dir = str(tmp_path)

    ctx = SessionContext(checkpoint_id=checkpoint_id, checkpoint_dir=checkpoint_dir)
    ctx.datastore["bad"] = object()

    mgr = CheckpointManager(checkpoint_id, checkpoint_dir=checkpoint_dir)
    try:
        mgr.save_state(
            {
                "metadata": {
                    "config": "dummy",
                    "checkpoint_id": checkpoint_id,
                    "game_id": "g",
                    "guid": "guid",
                    "frame_grids": [],
                    "available_actions": [],
                    "datastore": ctx.datastore_snapshot(),
                    "max_actions": 0,
                    "num_plays": 1,
                    "max_episode_actions": 0,
                    "action_counter": 0,
                    "current_play": 1,
                    "play_action_counter": 0,
                    "current_score": 0,
                    "current_state": "IN_PROGRESS",
                    "previous_score": 0,
                },
                "metrics": {
                    "total_cost": ctx.total_cost,
                    "total_usage": ctx.total_usage,
                    "action_history": ctx.action_history,
                },
            }
        )
        assert False, "Expected datastore_snapshot() to reject non-JSON-serializable values"
    except TypeError:
        pass


def test_context_last_frame_image_resize():
    """Test last_frame_image with resize parameter."""
    context = SessionContext()
    
    grids = [create_64x64_grid(1)]
    context.update(
        frame_grids=grids,
        current_score=0,
        current_state="IN_PROGRESS",
    )
    
    # Original size - check that image was created
    original = context.last_frame_image()
    assert original is not None
    original_size = original.size
    assert original_size[0] > 0 and original_size[1] > 0
    
    # Resize with int
    resized = context.last_frame_image(resize=32)
    assert resized is not None
    assert resized.size == (32, 32)
    
    # Resize with tuple
    resized_tuple = context.last_frame_image(resize=(16, 24))
    assert resized_tuple is not None
    assert resized_tuple.size == (16, 24)
    
    # Original should be unchanged
    assert context.last_frame_image().size == original_size


def test_context_datastore_access():
    """Test that datastore can be accessed and modified."""
    context = SessionContext()
    
    # Store values
    context.datastore["key1"] = "value1"
    context.datastore["key2"] = 42
    
    # Retrieve values
    assert context.datastore["key1"] == "value1"
    assert context.datastore["key2"] == 42
    
    # Use datastore methods - initialize counter first
    context.datastore["counter"] = 0
    context.datastore.increment("counter", 5)
    assert context.datastore["counter"] == 5


def test_context_update_with_empty_frames():
    """Test that update handles empty frame lists."""
    context = SessionContext()
    
    context.update(
        frame_grids=[],
        current_score=0,
        current_state="IN_PROGRESS",
    )
    
    assert len(context.frame_grids) == 0
    assert len(context.frame_images) == 0
    assert context.last_frame_image() is None
    assert context.last_frame_grid is None

