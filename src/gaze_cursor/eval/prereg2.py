"""Pre-registered round 2 (PROGRESS.md): H2 and H3. Every threshold here is the written protocol.

H2: calibrating in the lighting you will use beats a pool from other lighting.
H3: sessions that pass the brightness gate (face >= 60) are more accurate than sessions below it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gaze_cursor.config import ScreenConfig
from gaze_cursor.eval.configs import Config
from gaze_cursor.eval.session import Session
from gaze_cursor.eval.splits import Dataset, build_dataset, make_splits
from gaze_cursor.eval.stats import TargetStats, bootstrap_ci, bootstrap_diff_ci, concat, per_target

# Sessions that started before the round-2 pre-registration commit (3bb3fc3, local time) can never
# be test sessions; the two training twins are named explicitly.
PREREG2_AFTER = "2026-10-13T04:15:26"
F_TWIN = "session_20261013_040451.npz"   # front-lit, face 124 / background 79
B_TWIN = "session_20261013_033128.npz"   # backlit, face~78 / background~146 (from its note)
F_RANGE = (93.0, 155.0)                  # +-25% of 124
B_RANGE = (60.0, 97.0)                   # gate .. +25% of 78
B_RATIO_MAX = 0.6
D_RANGE = (25.0, 50.0)                   # below the gate, still trackable
N_DIM = 2
MIN_VALID = 0.8                          # objective-failure rule, as in H1
H2_MIN_GAIN_DEG, H3_MIN_GAP_DEG = 0.5, 1.5
N_BOOT, SEED = 10_000, 0


def lighting_class(meta: dict) -> str | None:
    """F / B / D from the captured lighting only (never from results), or None if outside all ranges."""
    li = meta.get("lighting")
    if not li:
        return None
    face, ratio, label = li["face"], li["ratio"], li["label"]
    if label == "front-lit" and F_RANGE[0] <= face <= F_RANGE[1]:
        return "F"
    if label == "backlit" and B_RANGE[0] <= face <= B_RANGE[1] and ratio < B_RATIO_MAX:
        return "B"
    if label == "dim" and D_RANGE[0] <= face < D_RANGE[1]:
        return "D"
    return None


def class_of(face: float, background: float) -> str | None:
    """Round-2 class (F / B / D) for a face and background brightness, or None if outside every range."""
    ratio = face / background if background else float("nan")
    from gaze_cursor.posture import lighting_label
    return lighting_class({"lighting": {"face": face, "ratio": ratio, "label": lighting_label(face, background)}})


@dataclass
class Selection:
    f_twin: Session | None
    b_twin: Session | None
    f2: Session | None = None
    b2: Session | None = None
    dim: list[Session] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)   # (session, reason)
    ignored: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def select_sessions(clean: list[Session]) -> Selection:
    """Pick the round-2 sessions by the pre-registered rules (earliest qualifying per class)."""
    by_name = {s.path.name: s for s in clean}
    sel = Selection(by_name.get(F_TWIN), by_name.get(B_TWIN))
    for twin, name in ((sel.f_twin, F_TWIN), (sel.b_twin, B_TWIN)):
        if twin is None:
            sel.missing.append(f"training twin {name}")
    picked: dict[str, list[Session]] = {"F": [], "B": [], "D": []}
    for s in clean:
        if s.layout != "v2" or str(s.meta.get("started", "")) < PREREG2_AFTER:
            continue
        cls = lighting_class(s.meta)
        if "lighting" not in s.meta:
            sel.excluded.append((s.path.name, "no captured lighting"))
        elif s.valid_fraction < MIN_VALID:
            sel.excluded.append((s.path.name, f"only {s.valid_fraction:.0%} valid frames"))
        elif cls is None:
            li = s.meta["lighting"]
            sel.excluded.append((s.path.name, f"lighting {li['label']} (face {li['face']:.0f}, "
                                              f"ratio {li['ratio']:.2f}) is outside every class range"))
        elif len(picked[cls]) >= (N_DIM if cls == "D" else 1):
            sel.ignored.append(f"{s.path.name} ({cls}: extra qualifying session)")
        else:
            picked[cls].append(s)
    sel.f2 = picked["F"][0] if picked["F"] else None
    sel.b2 = picked["B"][0] if picked["B"] else None
    sel.dim = picked["D"]
    if sel.f2 is None:
        sel.missing.append(f"F2: front-lit session, face {F_RANGE[0]:.0f}-{F_RANGE[1]:.0f}")
    if sel.b2 is None:
        sel.missing.append(f"B2: backlit session, face {B_RANGE[0]:.0f}-{B_RANGE[1]:.0f}, ratio < {B_RATIO_MAX}")
    if len(sel.dim) < N_DIM:
        sel.missing.append(f"{N_DIM - len(sel.dim)} more dim session(s): face {D_RANGE[0]:.0f}-{D_RANGE[1]:.0f}")
    return sel


def _fit_predict(ds: Dataset, config: Config, train: np.ndarray, test: np.ndarray,
                 screen: ScreenConfig) -> TargetStats:
    model = config.make_model(screen).fit(ds.x[train], ds.y[train])
    return per_target(model.predict(ds.x[test]), ds.y[test], ds.target[test], ds.session[test], screen)


def run_h2(sel: Selection, config: Config, screen: ScreenConfig) -> dict:
    """Train on the same-lighting twin vs the other-lighting session; test on all targets of F2 and B2."""
    ds = build_dataset([sel.f_twin, sel.b_twin, sel.f2, sel.b2], config.feature_set)
    f_tw, b_tw, f2, b2 = 0, 1, 2, 3
    same, other = [], []
    for test, twin, alt in ((f2, f_tw, b_tw), (b2, b_tw, f_tw)):
        rows = np.flatnonzero(ds.session == test)
        same.append(_fit_predict(ds, config, np.flatnonzero(ds.session == twin), rows, screen))
        other.append(_fit_predict(ds, config, np.flatnonzero(ds.session == alt), rows, screen))
    same_all, other_all = concat(same), concat(other)
    d = other_all.err_deg - same_all.err_deg               # positive: same-lighting calibration better
    lo, hi = bootstrap_ci(d, N_BOOT, SEED)
    per_session = [float(x.err_deg.mean() - y.err_deg.mean()) for x, y in zip(other, same)]
    crit = {
        "(a) CI lower bound > 0": lo > 0,
        f"(b) mean d >= {H2_MIN_GAIN_DEG} deg": float(d.mean()) >= H2_MIN_GAIN_DEG,
        "(c) d > 0 in both test sessions": all(x > 0 for x in per_session),
    }
    return {"n_targets": int(len(d)), "mean_d": float(d.mean()), "ci": (lo, hi),
            "per_session_d": per_session, "err_same": float(same_all.err_deg.mean()),
            "err_other": float(other_all.err_deg.mean()), "criteria": crit, "supported": all(crit.values())}


def run_h3(sel: Selection, config: Config, screen: ScreenConfig) -> dict:
    """Within-session error of gated sessions {F2, B2} vs below-gate sessions {D1, D2}."""
    sessions = [sel.f2, sel.b2, *sel.dim]
    ds = build_dataset(sessions, config.feature_set)
    stats = [_fit_predict(ds, config, sp.train, sp.test, screen) for sp in make_splits(ds, "within")]
    gated = concat(stats[:2]).err_deg
    below = concat(stats[2:]).err_deg
    lo, hi = bootstrap_diff_ci(below, gated, N_BOOT, SEED)
    means = [float(s.err_deg.mean()) for s in stats]
    pairs_ok = all(m_d > m_g for m_g in means[:2] for m_d in means[2:])
    gap = float(below.mean() - gated.mean())
    crit = {
        "(a) CI lower bound > 0": lo > 0,
        f"(b) gap >= {H3_MIN_GAP_DEG} deg": gap >= H3_MIN_GAP_DEG,
        "(c) every gated session beats every dim session (4 of 4 pairs)": pairs_ok,
    }
    return {"n_gated": int(len(gated)), "n_below": int(len(below)), "gap": gap, "ci": (lo, hi),
            "session_means": dict(zip([s.path.name for s in sessions], means)),
            "criteria": crit, "supported": all(crit.values())}
