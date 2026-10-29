"""Entraînement et évaluation des modèles de prédiction (§ 5.2, protocole)."""

from __future__ import annotations

import numpy as np

from spareparts.predict.dataset import HORIZON, PredictionData, tabular_features


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Rappel, précision, F1, aire sous la courbe précision-rappel et score de Brier."""
    from sklearn.metrics import average_precision_score, brier_score_loss

    y = y_true.ravel().astype(int)
    s = score.ravel()
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rappel": recall,
        "precision": precision,
        "f1": f1,
        "pr_auc": float(average_precision_score(y, s)) if y.any() else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(s, 0, 1))),
        "seuil": threshold,
        "positifs": int(y.sum()),
        "exemples": int(y.size),
    }


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Seuil maximisant le F1 sur l'ensemble de validation."""
    from sklearn.metrics import precision_recall_curve

    p, r, t = precision_recall_curve(y_true.ravel().astype(int), score.ravel())
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[int(np.argmax(f1[:-1]))]) if len(t) else 0.5


def split_by_group(data: PredictionData, val_groups, test_groups):
    """Découpage par simulation : aucune machine-année ne sert à la fois à apprendre et à tester."""
    val = np.isin(data.group, list(val_groups))
    test = np.isin(data.group, list(test_groups))
    return data.subset(~val & ~test), data.subset(val), data.subset(test)


def train_lightgbm(train: PredictionData, val: PredictionData, test: PredictionData) -> dict:
    """Un classifieur LightGBM par composant pour la panne dans les 24 heures."""
    import lightgbm as lgb

    Xtr, Xva, Xte = (tabular_features(d) for d in (train, val, test))
    ytr, yva, yte = (d.label_24h() for d in (train, val, test))
    K = ytr.shape[1]
    p_val, p_test = np.zeros(yva.shape), np.zeros(yte.shape)
    for k in range(K):
        pos = max(int(ytr[:, k].sum()), 1)
        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            scale_pos_weight=(len(ytr) - pos) / pos,
            verbose=-1,
        )
        clf.fit(Xtr, ytr[:, k])
        p_val[:, k] = clf.predict_proba(Xva)[:, 1]
        p_test[:, k] = clf.predict_proba(Xte)[:, 1]
    thr = best_threshold(yva, p_val)
    return classification_metrics(yte, p_test, thr)


def train_network(
    arch: str,
    train: PredictionData,
    val: PredictionData,
    test: PredictionData,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 1e-3,
    event_weight: float = 5.0,
    seed: int = 0,
    log=print,
):
    """Entraîne un réseau de survie et renvoie (modèle, métriques sur le test)."""
    import torch

    from spareparts.predict.models import ARCHITECTURES, failure_probability, survival_loss

    torch.manual_seed(seed)
    K = train.event_bin.shape[1]
    model = ARCHITECTURES[arch](train.seq.shape[2], train.static.shape[1], K, HORIZON)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def tensors(d: PredictionData):
        return (
            torch.from_numpy(d.seq),
            torch.from_numpy(d.static),
            torch.from_numpy(d.event_bin.astype(np.int64)),
            torch.from_numpy(d.observed.astype(np.int64)),
        )

    tr, va, te = tensors(train), tensors(val), tensors(test)

    def predict(t):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(t[0]), 4096):
                out.append(model(t[0][i : i + 4096], t[1][i : i + 4096]))
        return torch.cat(out)

    best, best_state, patience = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tr[0]))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            logits = model(tr[0][idx], tr[1][idx])
            loss = survival_loss(logits, tr[2][idx], tr[3][idx], event_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        val_loss = float(survival_loss(predict(va), va[2], va[3], event_weight))
        train_loss = total / len(order)
        log(f"{arch} époque {epoch + 1:2d}  perte {train_loss:.4f}  validation {val_loss:.4f}")
        if val_loss < best - 1e-4:
            best, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    model.load_state_dict(best_state)

    p_val = failure_probability(predict(va), 1).numpy()
    p_test = failure_probability(predict(te), 1).numpy()
    thr = best_threshold(val.label_24h(), p_val)
    metrics = classification_metrics(test.label_24h(), p_test, thr)
    metrics["c_index_7j"] = _concordance(test, failure_probability(predict(te), 7).numpy())
    return model, metrics


def _concordance(data: PredictionData, risk7: np.ndarray) -> float:
    """Indice de concordance entre le risque à 7 jours et le délai avant panne observé."""
    try:
        from lifelines.utils import concordance_index
    except ImportError:
        return float("nan")
    durations = np.where(data.event_bin >= 0, data.event_bin, data.observed).ravel()
    events = (data.event_bin >= 0).ravel()
    if not events.any():
        return float("nan")
    return float(concordance_index(durations, -risk7.ravel(), events))
