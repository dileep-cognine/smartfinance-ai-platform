from common.utils.file_utils import read_json, write_json


def test_json_round_trip(tmp_path) -> None:
    path = tmp_path / "value.json"
    write_json(path, {"ok": True})
    assert read_json(path) == {"ok": True}
