"""Entraînement et évaluation des modèles de prédiction (§ 5.2, protocole)."""

from __future__ import annotations

import numpy as np

from spareparts.predict.dataset import HORIZON, PredictionData, tabular_features


def classification_curve(y_true: np.ndarray, score: np.ndarray, bins: int = 10):
    """Courbe de fiabilité probabiliste : probabilité prédite contre fréquence observée.

    Chaque tranche regroupe les exemples dont la probabilité prédite tombe dans un intervalle de
    largeur égale. Un modèle calibré place ses points sur la diagonale. Renvoie les points
    ``[proba moyenne, fréquence observée, effectif]`` et l'erreur de calibration attendue (ECE),
    moyenne des écarts pondérée par l'effectif de chaque tranche.
    """
    y = y_true.ravel().astype(int)
    s = np.clip(score.ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    points, ece = [], 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        confidence = float(s[mask].mean())
        frequency = float(y[mask].mean())
        points.append([round(confidence, 4), round(frequency, 4), int(mask.sum())])
        ece += mask.sum() / y.size * abs(frequency - confidence)
    return points, float(ece)


def classification_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Rappel, précision, F1, aire sous la courbe précision-rappel, Brier et calibration."""
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
        "ece": classification_curve(y, s)[1],
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

    def fit_horizon(days: int):
        ytr, yva, yte = (d.label_days(days) for d in (train, val, test))
        p_val, p_test = np.zeros(yva.shape), np.zeros(yte.shape)
        for k in range(ytr.shape[1]):
            # Moins de 200 pannes par composant : arbres peu profonds, poids des pannes plafonné
            # et sous-échantillonnage, sans quoi le modèle apprend les exemples par cœur.
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=50,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                scale_pos_weight=10.0,
                random_state=0,
                deterministic=True,
                force_row_wise=True,
                verbose=-1,
            )
            clf.fit(Xtr, ytr[:, k])
            p_val[:, k] = clf.predict_proba(Xva)[:, 1]
            p_test[:, k] = clf.predict_proba(Xte)[:, 1]
        return classification_metrics(yte, p_test, best_threshold(yva, p_val)), yte, p_test

    metrics, _, _ = fit_horizon(1)
    week, y7, p7 = fit_horizon(7)
    week["calibration_7j"] = classification_curve(y7, p7)[0]
    metrics["a_7_jours"] = week
    return metrics


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

    from spareparts.predict.models import ARCHITECTURES, survival_loss

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

    tr, va = tensors(train), tensors(val)

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
            total += loss.item() * len(idx)
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
    return model, evaluate_network(model, val, test)


def evaluate_network(model, val: PredictionData, test: PredictionData) -> dict:
    """Métriques à 24 heures et à 7 jours ; seuils choisis sur la validation."""
    import torch

    from spareparts.predict.models import failure_probability

    def logits(d: PredictionData):
        model.eval()
        seq, static = torch.from_numpy(d.seq), torch.from_numpy(d.static)
        with torch.no_grad():
            return torch.cat(
                [model(seq[i : i + 4096], static[i : i + 4096]) for i in range(0, len(seq), 4096)]
            )

    lv, lt = logits(val), logits(test)
    out = {}
    for days in (1, 7):
        p_val = failure_probability(lv, days).numpy()
        p_test = failure_probability(lt, days).numpy()
        thr = best_threshold(val.label_days(days), p_val)
        out[days] = classification_metrics(test.label_days(days), p_test, thr)
    metrics = out[1]
    metrics["a_7_jours"] = out[7]
    metrics["c_index_7j"] = _concordance(test, failure_probability(lt, 7).numpy())
    metrics["calibration_7j"] = classification_curve(
        test.label_days(7), failure_probability(lt, 7).numpy()
    )[0]
    return metrics


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
