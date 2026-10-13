import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from xgboost import XGBClassifier

from src import config


def make_logreg(C=1.0, seed=config.SEED):
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def make_xgb(seed=config.SEED, **params):
    defaults = dict(
        objective="multi:softprob", num_class=config.NUM_CLASSES, tree_method="hist", subsample=0.8,
        colsample_bytree=0.8, eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    )  # fmt: skip
    return XGBClassifier(**(defaults | params))


def conv_block(c_in, c_out, k, pool=True):
    layers = [nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
    return layers + [nn.MaxPool1d(2) if pool else nn.AdaptiveAvgPool1d(1)]


def conv_trunk(in_channels=1):
    # (batch, C, 280) -> (batch, 128, 1)
    return nn.Sequential(*conv_block(in_channels, 32, 7), *conv_block(32, 64, 5), *conv_block(64, 128, 3, pool=False))


class ECGCNN(nn.Module):
    """The specified baseline: 3 conv blocks + MLP head on the raw beat."""

    def __init__(self, n_classes=config.NUM_CLASSES, in_channels=1):
        super().__init__()
        self.features = conv_trunk(in_channels)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )  # fmt: skip

    def forward(self, x, feats=None):  # feats ignored, kept so all models share one signature
        return self.classifier(self.features(x))


class HybridCNN(nn.Module):
    """Conv trunk on the beat + small MLP on the RR features.

    A single beat can't tell S from N; timing relative to the patient's rhythm can. The RR
    normalisation stats live in buffers so the checkpoint takes raw feature values.
    """

    def __init__(self, n_features=len(config.RR_FEATURES), n_classes=config.NUM_CLASSES, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.features = conv_trunk(in_channels)
        self.rr_branch = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128 + 32, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, n_classes)
        )
        self.register_buffer("feat_mean", torch.zeros(n_features))
        self.register_buffer("feat_std", torch.ones(n_features))

    def set_feature_stats(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6))

    def forward(self, x, feats=None):
        if feats is None:
            raise ValueError("HybridCNN needs RR features")
        z = (feats - self.feat_mean) / self.feat_std
        return self.classifier(torch.cat([self.features(x).flatten(1), self.rr_branch(z)], dim=1))


class SeedEnsemble(nn.Module):
    """Same model trained with different seeds. Returns log of the mean softmax,
    so callers can treat it like a single model."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.in_channels = getattr(members[0], "in_channels", 1)

    def forward(self, x, feats=None):
        probs = torch.stack([m(x, feats).softmax(1) for m in self.members]).mean(0)
        return probs.clamp_min(1e-12).log()


# hybrid_rel = hybrid with a 2nd input channel: beat minus the patient's running median beat
MODEL_KWARGS = {"cnn": {}, "hybrid": {}, "hybrid_rel": {"in_channels": 2}}


def build_model(model_type="cnn"):
    cls = ECGCNN if model_type == "cnn" else HybridCNN
    return cls(**MODEL_KWARGS[model_type])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_type_of(model):
    if isinstance(model, SeedEnsemble):
        return model_type_of(model.members[0])
    if isinstance(model, HybridCNN):
        return "hybrid_rel" if model.in_channels == 2 else "hybrid"
    return "cnn"


def save_cnn(model, path, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_members = len(model.members) if isinstance(model, SeedEnsemble) else 1
    meta = {**(metadata or {}), "model_type": model_type_of(model), "n_members": n_members}
    torch.save({"state_dict": model.state_dict(), "metadata": meta}, path)


def load_cnn(path=config.CNN_CHECKPOINT, device="cpu"):
    """Load any of our torch checkpoints (CNN, hybrid or ensemble) in eval mode."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("metadata", {})
    kind, n = meta.get("model_type", "cnn"), meta.get("n_members", 1)
    model = SeedEnsemble([build_model(kind) for _ in range(n)]) if n > 1 else build_model(kind)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), meta


def apply_calibration(logp, temperature=1.0, bias=None):
    """softmax(logp / T + b)"""
    z = np.asarray(logp, dtype=np.float64) / temperature
    if bias is not None:
        z = z + np.asarray(bias, dtype=np.float64)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def prediction_sets(probs, qhat):
    # Mondrian conformal: class k is in the set when 1 - p_k <= qhat_k
    return (1.0 - np.atleast_2d(probs)) <= np.asarray(qhat)
