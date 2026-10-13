import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import ECGCNN, HybridCNN, SeedEnsemble, build_model, save_cnn  # noqa: E402

# plausible RR feature stats so the random hybrids see sensible inputs
RR_MEAN = [300.0, 300.0, 1.0, 75.0, 300.0, 1.0, 1.0]
RR_STD = [60.0, 60.0, 0.2, 15.0, 50.0, 0.2, 0.2]

# Checkpoints are randomly initialised so the tests never depend on trained models.


@pytest.fixture
def cnn_checkpoint(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "cnn_test.pt"
    save_cnn(ECGCNN().eval(), path, {"trained_at": "test", "best_epoch": 0})
    return path


@pytest.fixture
def hybrid_checkpoint(tmp_path):
    torch.manual_seed(1)
    model = HybridCNN()
    model.set_feature_stats(RR_MEAN, RR_STD)
    path = tmp_path / "hybrid_test.pt"
    save_cnn(model.eval(), path, {"trained_at": "test", "best_epoch": 0})
    return path


@pytest.fixture
def hybrid_rel_ensemble_checkpoint(tmp_path):
    # same format as the shipped model, but with 2 members
    members = []
    for seed in (3, 4):
        torch.manual_seed(seed)
        m = build_model("hybrid_rel")
        m.set_feature_stats(RR_MEAN, RR_STD)
        members.append(m.eval())
    path = tmp_path / "hybrid_rel_ens.pt"
    save_cnn(SeedEnsemble(members).eval(), path, {"trained_at": "test"})
    return path
