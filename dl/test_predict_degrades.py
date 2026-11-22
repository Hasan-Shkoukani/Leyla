"""No-torch test: predict() must degrade gracefully. Run: python3 -m pytest dl/test_predict_degrades.py"""
import sys
from unittest import mock
from dl import config, predict as P


def test_degrades_without_torch_or_index(tmp_path):
    P._state.clear()
    with mock.patch.dict(sys.modules, {"torch": None, "open_clip": None}), \
            mock.patch.object(config, "INDEX_PATH", tmp_path / "nope.npz"), \
            mock.patch.object(config, "ARTIFACTS", tmp_path), \
            mock.patch.object(config, "LOG_PATH", tmp_path / "p.log"):
        r = P.predict(b"not an image")
    P._state.clear()
    assert r == {"matches": [], "confident": False}
    assert (tmp_path / "p.log").exists()


def test_rank_dedups_by_product():
    m = P.rank([0.9, 0.5, 0.95, 0.1], ["a", "b", "a", "c"])
    assert [x["product_id"] for x in m] == ["a", "b", "c"] and m[0]["score"] == 0.95
