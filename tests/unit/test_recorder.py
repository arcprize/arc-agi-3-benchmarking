import json
import os
import threading
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from agents.recorder import RECORDING_SUFFIX, Recorder


@pytest.mark.unit
class TestRecorderInitialization:
    @pytest.mark.parametrize(
        "prefix,filename,guid,expected_guid",
        [
            ("test-game.agent.50", None, None, None),
            ("test", "custom.game.agent.guid123.recording.jsonl", None, "guid123"),
            (
                "test",
                None,
                "12345678-1234-1234-1234-123456789012",
                "12345678-1234-1234-1234-123456789012",
            ),
        ],
    )
    def test_recorder_creation(
        self, temp_recordings_dir, prefix, filename, guid, expected_guid
    ):
        recorder = Recorder(prefix=prefix, filename=filename, guid=guid)

        assert recorder.prefix == prefix
        if expected_guid:
            assert recorder.guid == expected_guid
        else:
            assert recorder.guid is not None
            assert len(recorder.guid) > 0

        assert recorder.filename.endswith(RECORDING_SUFFIX)
        if filename:
            assert recorder.filename.endswith(filename)
        else:
            assert prefix in recorder.filename

    def test_recorder_directory_creation(self, temp_recordings_dir):
        _ = Recorder(prefix="test")  # Create recorder to trigger directory creation

        assert os.path.exists(temp_recordings_dir)
        assert os.path.isdir(temp_recordings_dir)

    def test_recorder_with_empty_recordings_dir(self):
        with patch.dict("os.environ", {"RECORDINGS_DIR": ""}):
            recorder = Recorder(prefix="test")

            assert recorder.prefix == "test"
            assert recorder.guid is not None


@pytest.mark.unit
class TestRecorderFileOperations:
    def test_record_and_get_events(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-record")

        test_data = {"action": "RESET", "game_id": "test-game", "score": 10}
        recorder.record(test_data)

        assert os.path.exists(recorder.filename)

        with open(recorder.filename, "r") as f:
            line = f.readline().strip()
            event = json.loads(line)

            assert "timestamp" in event
            assert "data" in event
            assert event["data"] == test_data

        events_data = [
            {"action": "ACTION1", "score": 5},
            {"action": "ACTION2", "score": 10},
        ]

        for data in events_data:
            recorder.record(data)

        recorded_events = recorder.get()
        assert len(recorded_events) == 3

        assert recorded_events[0]["data"] == test_data
        assert recorded_events[1]["data"] == events_data[0]
        assert recorded_events[2]["data"] == events_data[1]

        for event in recorded_events:
            assert "timestamp" in event

    def test_record_with_complex_data(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-complex")

        complex_data = {
            "action": "ACTION6",
            "coordinates": {"x": 32, "y": 15},
            "frame": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]],
            "reasoning": {"model": "o4-mini", "tokens": 150, "confidence": 0.85},
            "metadata": ["tag1", "tag2", "tag3"],
        }

        recorder.record(complex_data)

        recorded_events = recorder.get()
        assert len(recorded_events) == 1
        assert recorded_events[0]["data"] == complex_data

    def test_get_events_empty_file(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-empty")

        events = recorder.get()
        assert events == []

    def test_get_events_with_invalid_json(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-invalid")

        with open(recorder.filename, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("invalid json line\n")
            f.write('{"another": "valid"}\n')

        with pytest.raises(json.JSONDecodeError):
            recorder.get()


@pytest.mark.unit
class TestRecorderTimestamps:
    def test_timestamp_format_and_sequence(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-timestamp")

        recorder.record({"event": 1})
        events = recorder.get()
        timestamp_str = events[0]["timestamp"]

        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)

        now = datetime.now(timestamp.tzinfo)
        time_diff = abs((now - timestamp).total_seconds())
        assert time_diff < 60

        time.sleep(0.001)
        recorder.record({"event": 2})
        time.sleep(0.001)
        recorder.record({"event": 3})

        events = recorder.get()

        timestamps = [
            datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00"))
            for e in events
        ]

        assert timestamps[0] <= timestamps[1] <= timestamps[2]


@pytest.mark.unit
class TestRecorderClassMethods:
    def test_list_recordings(self, temp_recordings_dir):
        import glob

        for f in glob.glob(os.path.join(temp_recordings_dir, "*")):
            os.unlink(f)

        test_files = [
            "game1.agent.50.guid1.recording.jsonl",
            "game2.agent.25.guid2.recording.jsonl",
            "not-a-recording.txt",
        ]

        for filename in test_files:
            filepath = os.path.join(temp_recordings_dir, filename)
            with open(filepath, "w") as f:
                f.write('{"test": "data"}\n')

        with patch.dict("os.environ", {"RECORDINGS_DIR": temp_recordings_dir}):
            recordings = Recorder.list()

        recording_jsonl_files = [
            f for f in recordings if f.endswith(".recording.jsonl")
        ]
        assert len(recording_jsonl_files) == 2

        assert "not-a-recording.txt" not in recordings
        assert any("game1.agent.50.guid1.recording.jsonl" in f for f in recordings)
        assert any("game2.agent.25.guid2.recording.jsonl" in f for f in recordings)

    def test_list_recordings_empty_dir(self, temp_recordings_dir):
        import glob

        for f in glob.glob(os.path.join(temp_recordings_dir, "*.recording.jsonl")):
            os.unlink(f)

        with patch.dict("os.environ", {"RECORDINGS_DIR": temp_recordings_dir}):
            recordings = Recorder.list()
        assert recordings == []

    @pytest.mark.parametrize(
        "filename,expected_prefix,expected_prefix_one,expected_guid",
        [
            (
                "locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl",
                "locksmith.random.50",
                "locksmith",
                "81329339-1951-487c-8bed-e9d4780320f2",
            ),
            ("a.b.c.recording.jsonl", "a.b", "a", "c"),
            ("simple", "simple", "simple", "simple"),
        ],
    )
    def test_filename_parsing(
        self, filename, expected_prefix, expected_prefix_one, expected_guid
    ):
        assert Recorder.get_prefix(filename) == expected_prefix
        assert Recorder.get_prefix_one(filename) == expected_prefix_one
        assert Recorder.get_guid(filename) == expected_guid


@pytest.mark.unit
class TestRecorderErrorHandling:
    def test_recording_to_readonly_file(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-readonly")

        with open(recorder.filename, "w") as f:
            f.write('{"initial": "data"}\n')

        os.chmod(recorder.filename, 0o444)

        try:
            with pytest.raises(PermissionError):
                recorder.record({"test": "data"})
        finally:
            os.chmod(recorder.filename, 0o644)

    def test_recording_large_data(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-large")

        large_data = {
            "action": "TEST",
            "large_array": [[i * j for i in range(100)] for j in range(100)],
            "metadata": {"description": "x" * 1000},
        }

        recorder.record(large_data)

        events = recorder.get()
        assert len(events) == 1
        assert events[0]["data"]["action"] == "TEST"
        assert len(events[0]["data"]["large_array"]) == 100

    def test_unicode_data_handling(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-unicode")

        unicode_data = {
            "action": "TEST",
            "text": "Hello 世界! 🎮🤖",
            "symbols": "αβγδε",
            "emoji": "🎯🚀💡",
        }

        recorder.record(unicode_data)

        events = recorder.get()
        assert len(events) == 1
        assert events[0]["data"]["text"] == "Hello 世界! 🎮🤖"
        assert events[0]["data"]["emoji"] == "🎯🚀💡"

    def test_concurrent_recording(self, temp_recordings_dir):
        recorder = Recorder(prefix="test-concurrent")

        results = []
        errors = []

        def record_data(data):
            try:
                recorder.record(data)
                results.append(data)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=record_data, args=({"thread": i, "data": f"test-{i}"},)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 5

        events = recorder.get()
        assert len(events) == 5
