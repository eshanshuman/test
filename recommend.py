"""recommend.py - Hybrid recommendation engine (rules + trained model).

Purpose
    Turns one set of readings into a status and a prioritised to-do list.
      1. RULES: each parameter is graded against the P. vannamei bands in
         config.py and mapped to a management action with an urgency
         (immediate / today / this week / routine) and a reason. The rules are
         the part a farmer and an examiner can read and check.
      2. MODEL: the trained LightGBM (else Random Forest) from
         models/<dataset>/ gives a second opinion and a confidence.
      3. HYBRID: if rules and model agree, that is the status. If they
         disagree, the MORE SEVERE status is used (missing a real problem
         costs a crop; a false alarm costs a walk to the pond) and "Verify the
         probe readings" is added, because disagreement usually means a
         reading sits on a threshold or a probe is drifting.

    Doses and percentages are typical values from Indian extension guidance
    for vannamei culture (CIBA / MPEDA advisories) as recalled by the author
    of this code; they have NOT been checked against the source documents and
    every action text tells the farmer to confirm with a local soil and water
    test. They are collected in DOSES below so they can be corrected in one
    place.

Paper link
    The "recommendation" layer of the Aquabot brief; the reviewed papers stop
    at prediction (Chen et al. 2022; Hemal et al. 2024; Huang & Khabusi 2025).

Outputs (when run as a script)
    figures/<dataset>/fig17_surrogate_tree.png (depth-3 tree that imitates
    the model); reports/<dataset>/recommend_scenarios.csv (4 worked
    scenarios); block "recommend" in metrics.json.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import config as cfg

URGENCY = ["immediate", "today", "this week", "routine"]          # priority order
CONFIRM = "Confirm the dose with a local soil and water test before applying."
# Typical extension-guidance figures. UNVERIFIED against CIBA/MPEDA documents: check before use.
DOSES = {
    "lime": "agricultural lime or dolomite at about 100-200 kg/ha, split over 2-3 days",
    "lime_light": "dolomite at about 50-100 kg/ha",
    "exchange_high_ph": "15-25 %",
    "exchange_light": "about 10 %",
    "salinity_step": "no more than 2-3 ppt per day",
}
PRESETS = {      # also the dashboard presets: ph, temperature_c, do_mg_l, salinity_ppt
    "Healthy afternoon": (8.1, 30.0, 7.5, 18.0),
    "Dawn hypoxia": (7.6, 28.5, 2.4, 18.0),
    "Acid rain event": (6.8, 27.5, 5.5, 8.0),
    "Heatwave + bloom": (9.2, 34.6, 11.0, 24.0),
}


def _act(urgency: str, parameter: str, action: str, reason: str) -> dict:
    return {"urgency": urgency, "parameter": parameter, "action": action, "reason": reason}


def rule_actions(ph: float, temperature_c: float, do_mg_l: float, salinity_ppt: float) -> list[dict]:
    """One or more actions per parameter that is outside its Good band."""
    a = []
    # --- dissolved oxygen ----------------------------------------------------
    if do_mg_l < cfg.DO_ANOXIA:
        a.append(_act("immediate", "DO", "Run every aerator now and stop feeding until DO is back above 4 mg/L.",
                      f"DO {do_mg_l:.1f} mg/L is below 2: prawns suffocate within hours."))
    elif do_mg_l < 3:
        a.append(_act("immediate", "DO", "Start emergency aeration and halve the next feed.",
                      f"DO {do_mg_l:.1f} mg/L is below 3 (Critical): feeding adds oxygen demand."))
    elif do_mg_l < 5:
        a.append(_act("today", "DO", "Extend night aeration through to after sunrise and check feed trays for leftovers.",
                      f"DO {do_mg_l:.1f} mg/L is below 5 (Moderate): the dawn minimum will be lower still."))
    # --- pH ---------------------------------------------------------------------
    if ph < 7.0:
        a.append(_act("today", "pH", f"Apply {DOSES['lime']}. {CONFIRM}",
                      f"pH {ph:.2f} is below 7 (Critical): acid water stresses prawns and slows moulting."))
    elif ph < 7.5:
        a.append(_act("this week", "pH", f"Apply {DOSES['lime_light']} and re-check morning and afternoon pH. {CONFIRM}",
                      f"pH {ph:.2f} is below 7.5 (Moderate): low buffering, often after rain."))
    elif ph > 9.0:
        a.append(_act("today", "pH", f"Exchange {DOSES['exchange_high_ph']} of the water and stop liming.",
                      f"pH {ph:.2f} is above 9 (Critical): ammonia becomes far more toxic at high pH."))
    elif ph > 8.5:
        a.append(_act("this week", "pH", f"Hold back lime and fertiliser; exchange {DOSES['exchange_light']} of the water if the afternoon pH keeps rising.",
                      f"pH {ph:.2f} is above 8.5 (Moderate): usually a dense algal bloom."))
    # --- temperature -----------------------------------------------------------
    if temperature_c > 34.0:
        a.append(_act("today", "Temperature", "Raise the water level, shade part of the pond, and feed only at dawn and dusk.",
                      f"{temperature_c:.1f} °C is above 34 (Critical): warm water holds less oxygen while prawns need more."))
    elif temperature_c > 32.0:
        a.append(_act("this week", "Temperature", "Move the main feeds to early morning and evening and keep the water level up.",
                      f"{temperature_c:.1f} °C is above 32 (Moderate)."))
    elif temperature_c < 26.0:
        a.append(_act("today", "Temperature", "Cut the ration to what the trays show is eaten and avoid water exchange during the cold spell.",
                      f"{temperature_c:.1f} °C is below 26 (Critical): feeding and growth slow sharply."))
    elif temperature_c < 28.0:
        a.append(_act("this week", "Temperature", "Reduce the ration slightly and check feed trays after each feed.",
                      f"{temperature_c:.1f} °C is below 28 (Moderate): appetite drops in cool water."))
    # --- salinity ---------------------------------------------------------------
    if salinity_ppt < 5.0:
        a.append(_act("today", "Salinity", f"Raise salinity with saline water, changing {DOSES['salinity_step']}.",
                      f"{salinity_ppt:.1f} ppt is below 5 (Critical): osmotic stress and mineral shortage."))
    elif salinity_ppt > 35.0:
        a.append(_act("today", "Salinity", f"Lower salinity with fresh water, changing {DOSES['salinity_step']}.",
                      f"{salinity_ppt:.1f} ppt is above 35 (Critical): osmotic stress and slow growth."))
    elif salinity_ppt < 10.0 or salinity_ppt > 25.0:
        way = "saline" if salinity_ppt < 10.0 else "fresh"
        a.append(_act("this week", "Salinity", f"Bring salinity back towards 10-25 ppt with {way} water, {DOSES['salinity_step']}.",
                      f"{salinity_ppt:.1f} ppt is outside 10-25 (Moderate)."))
    return a


def feeding_action(status: str, do_mg_l: float) -> dict:
    if do_mg_l < cfg.DO_ANOXIA:
        return _act("immediate", "Feeding", "No feed until DO has recovered.", "Uneaten feed and digestion both consume oxygen.")
    if status == "Critical":
        return _act("today", "Feeding", "Halve the ration and feed only what the trays show is eaten.",
                    "Stressed prawns eat less; leftover feed fouls the water.")
    if status == "Moderate":
        return _act("today", "Feeding", "Reduce the ration and check feed trays 2 hours after each feed.",
                    "Conditions are outside the Good band, so appetite is likely down.")
    return _act("routine", "Feeding", "Feed the normal ration and keep the routine tray checks.",
                "All four parameters are in the Good band.")


@lru_cache(maxsize=1)
def load_model():
    """LightGBM if it was trained, else Random Forest, else None."""
    import joblib
    for name, fname in (("LightGBM", "lightgbm.joblib"), ("Random Forest", "random_forest.joblib")):
        path = cfg.MODEL_DIR / fname
        if path.exists():
            return name, joblib.load(path)
    return None, None


def recommend(ph: float, temperature_c: float, do_mg_l: float, salinity_ppt: float) -> dict:
    """Status + prioritised actions for one reading. See the module docstring."""
    rule_status = cfg.label_reading(ph, temperature_c, do_mg_l, salinity_ppt)
    model_name, model = load_model()
    model_status = confidence = None
    if model is not None:
        row = pd.DataFrame([[ph, temperature_c, do_mg_l, salinity_ppt]], columns=cfg.FEATURES)
        proba = model.predict_proba(row)[0]
        best = int(np.argmax(proba))
        model_status, confidence = cfg.CLASSES[int(model.classes_[best])], float(proba[best])
    agree = None if model_status is None else (model_status == rule_status)
    status = rule_status
    if agree is False:                                    # take the more severe of the two
        status = max(rule_status, model_status, key=cfg.CLASSES.index)
    actions = rule_actions(ph, temperature_c, do_mg_l, salinity_ppt)
    if agree is False:
        actions.append(_act("today", "Sensors", "Verify the probe readings: re-measure with a handheld meter and check the last calibration date.",
                            f"Rules say {rule_status} but the {model_name} model says {model_status} "
                            f"({confidence:.0%} confident); a reading is near a threshold or a probe has drifted."))
    actions.append(feeding_action(status, do_mg_l))
    actions.sort(key=lambda x: URGENCY.index(x["urgency"]))          # stable: keeps parameter order
    return {"status": status, "rule_status": rule_status, "model_status": model_status,
            "model_name": model_name, "model_confidence": confidence, "agreement": agree,
            "actions": actions,
            "n_immediate": sum(x["urgency"] == "immediate" for x in actions)}


def surrogate_tree_figure() -> float | None:
    """Depth-3 tree trained to imitate the model; returns its fidelity on the test rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    model_name, model = load_model()
    if model is None:
        print("  no trained model in models/ - run 03 or 04 first; figure skipped")
        return None
    cfg.apply_style()
    X_train, X_test, _, _ = cfg.classification_split()
    tree = DecisionTreeClassifier(max_depth=3, random_state=cfg.SEED).fit(X_train, model.predict(X_train))
    fidelity = float((tree.predict(X_test) == model.predict(X_test)).mean())
    fig, ax = plt.subplots(figsize=(13, 5.6))
    names = [cfg.CLASSES[int(c)] for c in tree.classes_]
    boxes = plot_tree(tree, feature_names=[cfg.PRETTY[c] for c in cfg.FEATURES], class_names=names,
                      filled=True, impurity=False, rounded=True, fontsize=8, ax=ax, precision=2)
    for box in boxes:                                     # recolour nodes with the status palette
        patch = box.get_bbox_patch()
        majority = box.get_text().rsplit("class = ", 1)[-1].strip()
        if patch is not None and majority in cfg.STATUS_COLOURS:
            rgb = np.array(matplotlib.colors.to_rgb(cfg.STATUS_COLOURS[majority]))
            patch.set_facecolor(0.5 * rgb + 0.5)          # tint towards white, opaque so arrows stay behind
            patch.set_alpha(1.0)
            patch.set_edgecolor("#333333")
    ax.grid(False)
    ax.set_title(f"Depth-3 tree imitating the {model_name} model (agrees with it on {100 * fidelity:.1f} % "
                 f"of test readings)\n{cfg.DATASET} dataset ({cfg.DATASET_NOTE}); box colour = majority status")
    cfg.save_fig(fig, "fig17_surrogate_tree")
    return fidelity


def main() -> None:
    cfg.section(f"Recommendation engine on the {cfg.DATASET} dataset ({cfg.DATASET_NOTE})")
    model_name, _ = load_model()
    print(f"  model in use: {model_name or 'none (rules only)'}")
    rows = []
    for name, reading in PRESETS.items():
        r = recommend(*reading)
        cfg.section(f"Scenario: {name}")
        print("  readings : pH {:.1f}, {:.1f} °C, DO {:.1f} mg/L, salinity {:.0f} ppt".format(*reading))
        conf = "" if r["model_confidence"] is None else f" ({r['model_confidence']:.0%} confident)"
        print(f"  STATUS   : {r['status']}   [rules: {r['rule_status']}; model: {r['model_status']}{conf}; "
              f"agree: {r['agreement']}]")
        for act in r["actions"]:
            print(f"  [{act['urgency']:<9}] {act['parameter']}: {act['action']}")
            print(f"  {'':<12}why: {act['reason']}")
        rows.append({"scenario": name, "ph": reading[0], "temperature_c": reading[1], "do_mg_l": reading[2],
                     "salinity_ppt": reading[3], "status": r["status"], "rule_status": r["rule_status"],
                     "model_status": r["model_status"], "model_confidence": r["model_confidence"],
                     "agreement": r["agreement"], "n_actions": len(r["actions"]),
                     "n_immediate": r["n_immediate"]})
    table = pd.DataFrame(rows).round(4)
    table.to_csv(cfg.REPORT_DIR / "recommend_scenarios.csv", index=False)

    cfg.section("Surrogate decision tree")
    fidelity = surrogate_tree_figure()
    if fidelity is not None:
        print(f"  the depth-3 tree reproduces the model's answer on {100 * fidelity:.1f} % of test readings")
    cfg.record_metric("recommend", {"model": model_name,
                                    "surrogate_fidelity": None if fidelity is None else round(fidelity, 4),
                                    "scenarios": table.to_dict(orient="records")})


if __name__ == "__main__":
    main()
