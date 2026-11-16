"""Weights load from a package-relative file, and a pickle payload does not run."""

from pathlib import Path

import pytest
import torch

import src.microstructure as ms


def _touch_sentinel(path):
    Path(path).write_text("pwned")


@pytest.fixture
def fresh_model(monkeypatch):
    """Undo the autouse patch that forces _model_trained to True."""
    ms._model = None
    ms._device = None
    monkeypatch.setattr(ms, "_model_trained", False)
    yield
    ms._model = None
    ms._device = None
    monkeypatch.setattr(ms, "_model_trained", False)


def test_state_dict_sets_trained_flag(tmp_path, fresh_model, monkeypatch):
    path = tmp_path / "orderbook_cnn.pt"
    torch.save(ms.OrderBookCNN().state_dict(), path)
    monkeypatch.setattr(ms, "MODEL_PATH", path)
    monkeypatch.setattr("src.market.get_intraday_data", lambda ticker: None)
    ms.analyze_liquidity("TEST")
    assert ms._model_trained is True


def test_pickle_payload_does_not_run(tmp_path, fresh_model, monkeypatch):
    sentinel = tmp_path / "sentinel.txt"

    class Evil:
        def __reduce__(self):
            return (_touch_sentinel, (str(sentinel),))

    path = tmp_path / "evil.pt"
    import pickle
    path.write_bytes(pickle.dumps(Evil()))
    monkeypatch.setattr(ms, "MODEL_PATH", path)
    with pytest.raises(Exception):
        ms._get_model()
    assert not sentinel.exists()
