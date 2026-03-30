import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

RECORDING_SUFFIX = ".recording.jsonl"


def get_recordings_dir() -> str:
    """Get the current recordings directory from environment variable."""
    return os.environ.get("RECORDINGS_DIR", "")


class Recorder:
    def __init__(
        self, prefix: str, filename: Optional[str] = None, guid: Optional[str] = None
    ) -> None:
        self.guid = self.get_guid(filename) if filename else (guid or str(uuid.uuid4()))
        self.prefix: str = prefix
        recordings_dir = get_recordings_dir()
        self.filename = (
            os.path.join(recordings_dir, filename)
            if filename
            else os.path.join(
                recordings_dir,
                f"{self.prefix}.{self.guid}{RECORDING_SUFFIX}",
            )
        )
        # Create directory once during initialization
        if recordings_dir:
            os.makedirs(recordings_dir, exist_ok=True)

    def record(self, data: dict[str, Any]) -> None:
        """
        Records an event to the file.
        `data` should be a dictionary (JSON-serializable) or a JSON string.
        """
        event: dict[str, Any] = {}
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["data"] = data

        with open(self.filename, "a", encoding="utf-8") as f:
            json.dump(event, f)
            f.write("\n")

    def get(self) -> list[dict[str, Any]]:
        """
        Loads all recorded events and returns them as a list of dictionaries.
        """
        if not os.path.isfile(self.filename):
            return []

        events: list[dict[str, Any]] = []
        with open(self.filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def __repr__(self) -> str:
        return f"<Recorder guid={self.guid} file={self.filename}>"

    @classmethod
    def list(cls) -> list[str]:
        recordings_dir = get_recordings_dir()
        if recordings_dir:
            os.makedirs(recordings_dir, exist_ok=True)
            filenames = os.listdir(recordings_dir)
        else:
            filenames = []
        return [f for f in filenames if f.endswith(RECORDING_SUFFIX)]

    @classmethod
    def get_prefix(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: locksmith.random.50
        """
        if "." in filename:
            parts = filename.split(".")
            return ".".join(parts[:-3])
        else:
            return filename

    @classmethod
    def get_prefix_one(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: locksmith
        """
        if "." in filename:
            parts = filename.split(".")
            return parts[0]
        else:
            return filename

    @classmethod
    def get_guid(cls, filename: str) -> str:
        """
        Example filename: locksmith.random.50.81329339-1951-487c-8bed-e9d4780320f2.recording.jsonl
        Returns: 81329339-1951-487c-8bed-e9d4780320f2
        """
        if "." in filename:
            parts = filename.split(".")
            return parts[-3]
        else:
            return filename
