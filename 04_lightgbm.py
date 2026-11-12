"""04 - LightGBM classifier for pond status, with exact tree-SHAP explanations.

Purpose
    A gradient-boosted alternative to the Random Forest, on exactly the same
    train/test split (config.classification_split) so the two are comparable.
    Part of the training rows is held back as a validation set, and boosting
    stops when validation loss stops improving (early stopping), which picks
    the number of trees without ever looking at the test set.

    Explanations: LightGBM computes exact tree-SHAP values itself
    (predict(pred_contrib=True)). For each prediction they split the model's
    score into one contribution per sensor reading; the script checks that the
    contributions add up to the model's raw score, then plots the mean
    absolute contribution of each parameter to each class.

Paper link
    Gradient boosting is the stronger tabular model in the reviewed
    IoT-aquaculture literature (Hemal et al. 2024; Huang & Khabusi 2025). The
    brief's "2026 LightGBM paper" was not supplied and is not reproduced.

Outputs
    figures/<dataset>/fig09_lgbm_shap.png, fig10_lgbm_confusion_matrix.png;
    models/<dataset>/lightgbm.joblib; reports/<dataset>/04_lgbm_classification_report.csv;
    block "04_lightgbm" in metrics.json.
"""
from __future__ import annotations

import inspect
import json

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import config as cfg

cfg.apply_style()
NOTE = f"{cfg.DATASET} dataset ({cfg.DATASET_NOTE})"
LABELS = list(range(len(cfg.CLASSES)))
MAX_TREES, PATIENCE = 2000, 50


def scores(y_true, y_pred) -> dict:
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                                             zero_division=0)), 4)}


def shap_by_class(model, X_test) -> pd.DataFrame:
    """Mean |SHAP| per parameter and class, after checking the values are exact."""
    n_feat, n_class = len(cfg.FEATURES), len(model.classes_)
    contrib = model.booster_.predict(X_test, pred_contrib=True)
    contrib = np.asarray(contrib).reshape(len(X_test), n_class, n_feat + 1)   # last = base value
    raw = model.booster_.predict(X_test, raw_score=True).reshape(len(X_test), n_class)
    worst = float(np.abs(contrib.sum(axis=2) - raw).max())
    print(f"  additivity check: contributions + base value equal the raw score "
          f"(largest difference {worst:.1e})")
    assert worst < 1e-6, "SHAP contributions do not add up to the model output"
    mean_abs = np.abs(contrib[:, :, :n_feat]).mean(axis=0)                     # class x feature
    return pd.DataFrame(mean_abs.T, index=cfg.FEATURES,
                        columns=[cfg.CLASSES[c] for c in model.classes_])


def fig_shap(shap: pd.DataFrame) -> None:
    order = shap.sum(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(7, 3.6))
    height = 0.8 / shap.shape[1]
    for k, name in enumerate(shap.columns):
        ypos = np.arange(len(order)) + (k - (shap.shape[1] - 1) / 2) * height
        ax.barh(ypos, shap.loc[order, name], height=height * 0.9,
                color=cfg.STATUS_COLOURS[name], label=f"towards/away from {name}")
    ax.set_yticks(range(len(order)), [cfg.PRETTY[c] for c in order])
    ax.set_xlabel("Mean |SHAP| on the test set (log-odds units)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"LightGBM: how much each reading moves each class score\n{NOTE}")
    cfg.save_fig(fig, "fig09_lgbm_shap")


def main() -> None:
    cfg.section(f"04 LightGBM on the {NOTE}")
    X_train, X_test, y_train, y_test = cfg.classification_split()
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15,
                                                  stratify=y_train, random_state=cfg.SEED)
    print(f"  same split as 03: train {len(X_train):,} / test {len(X_test):,}")
    print(f"  training rows split again for early stopping: fit {len(X_fit):,} / validation {len(X_val):,}")

    cfg.section("Training with early stopping")
    model = lgb.LGBMClassifier(n_estimators=MAX_TREES, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=1.0,
                               random_state=cfg.SEED, n_jobs=-1, verbose=-1)
    # newer LightGBM takes eval_X/eval_y; older releases only know eval_set
    new_api = "eval_X" in inspect.signature(model.fit).parameters
    eval_kw = {"eval_X": X_val, "eval_y": y_val} if new_api else {"eval_set": [(X_val, y_val)]}
    model.fit(X_fit, y_fit, eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)], **eval_kw)
    val_loss = float(next(iter(model.best_score_.values()))["multi_logloss"])
    print(f"  stopped at {model.best_iteration_} trees (limit {MAX_TREES}, patience {PATIENCE}); "
          f"validation log-loss {val_loss:.4f}")

    cfg.section("Held-out test set")
    pred = model.predict(X_test)
    lgbm = scores(y_test, pred)
    rule = scores(y_test, cfg.label_frame(X_test).map({c: i for i, c in enumerate(cfg.CLASSES)}))
    rows = [("LightGBM", lgbm), ("Labelling rule on same readings", rule)]
    prev = json.loads(cfg.METRICS_JSON.read_text(encoding="utf-8")) if cfg.METRICS_JSON.exists() else {}
    if "03_random_forest" in prev:
        rows.insert(1, ("Random Forest (from 03)", prev["03_random_forest"]["test"]))
    print(f"  {'method':<34}{'accuracy':>10}{'macro F1':>10}")
    for name, s in rows:
        print(f"  {name:<34}{s['accuracy']:>10.4f}{s['macro_f1']:>10.4f}")
    gap = lgbm["accuracy"] - rule["accuracy"]
    print(f"\n  LightGBM minus rule reference: {gap:+.4f} accuracy")

    report = classification_report(y_test, pred, labels=LABELS, target_names=cfg.CLASSES,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.round(4)
    print("\n" + report_df.loc[cfg.CLASSES, ["precision", "recall", "f1-score", "support"]].to_string())
    report_df.to_csv(cfg.REPORT_DIR / "04_lgbm_classification_report.csv")

    cfg.section("Exact tree-SHAP (pred_contrib=True)")
    shap = shap_by_class(model, X_test)
    print(shap.rename(index=cfg.PRETTY).round(3).to_string())

    cfg.section("Figures and model")
    fig_shap(shap)
    cm = confusion_matrix(y_test, pred, labels=LABELS)
    cfg.plot_confusion(cm, cfg.CLASSES, f"LightGBM: test confusion matrix\n{NOTE}",
                       "fig10_lgbm_confusion_matrix")
    joblib.dump(model, cfg.MODEL_DIR / "lightgbm.joblib", compress=3)
    print(f"  saved {(cfg.MODEL_DIR / 'lightgbm.joblib').relative_to(cfg.ROOT).as_posix()}")

    cfg.record_metric("04_lightgbm", {
        "n_fit": len(X_fit), "n_val": len(X_val), "n_test": len(X_test),
        "best_iteration": int(model.best_iteration_), "val_logloss": round(val_loss, 4),
        "test": lgbm, "rule_reference": rule, "accuracy_minus_rule": round(gap, 4),
        "per_class_f1": {c: report[c]["f1-score"] for c in cfg.CLASSES},
        "mean_abs_shap": shap.round(4).to_dict(), "confusion_matrix": cm})


if __name__ == "__main__":
    main()
