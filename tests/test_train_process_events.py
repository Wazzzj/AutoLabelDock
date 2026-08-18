import json

from src.engine.train_process import EVENT_PREFIX, _emit


def test_training_event_can_be_written_without_stdout(tmp_path):
    event_path = tmp_path / "events.jsonl"

    _emit("epoch", {"epoch": 1}, event_path)

    line = event_path.read_text(encoding="utf-8").strip()
    assert line.startswith(EVENT_PREFIX)
    assert json.loads(line[len(EVENT_PREFIX):]) == {
        "type": "epoch",
        "payload": {"epoch": 1},
    }
