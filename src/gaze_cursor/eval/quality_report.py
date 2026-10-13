"""Turn session-quality numbers into pass / warn / fail lines (M3 step 6b). Pure functions.

The cut-offs below are the ones written in PLAN.md (M3 step 6b, "6b thresholds"), set from the
existing sessions BEFORE this module existed. If one changes after seeing a new session's result,
log the change and the reason in PLAN.md; do not tune them quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PASS, WARN, FAIL, NA, INFO = "pass", "warn", "fail", "n/a", "info"   # info = reported, never scored
_RANK = {NA: -1, INFO: -1, PASS: 0, WARN: 1, FAIL: 2}

# (pass bound, warn bound): lower is better for "max" bands, higher is better for "min" bands.
T = {
    "extent_x": (0.90, 0.80), "extent_y": (0.85, 0.80), "outside_hull": (0.0, 0.05),
    "valid_fraction": (0.95, 0.90), "thin_targets": (0.0, 1.0), "dropped_gap_share": (0.02, 0.05),
    "heldout_mean": (3.5, 5.0), "loto_mean": (4.5, 6.0),
    "far_share": (0.10, 0.30),                      # strict: share < bound
    "slope_x_warn": 0.85, "slope_y_warn": 0.25, "spread_warn": 2.5,
    "head_shift_cm": (1.0, 2.0), "head_rot_deg": (3.0, 5.0),
    "face_brightness": (100.0, 60.0),
}
TITLES = {1: "Coverage", 2: "Fixation quality", 3: "Held-out fit", 4: "Bias vs noise (warn-only)",
          5: "Drift and head", 6: "Conditions"}


@dataclass(frozen=True)
class Line:
    name: str
    value: float | str
    level: str
    note: str = ""


@dataclass(frozen=True)
class CheckResult:
    check: int
    title: str
    lines: list[Line] = field(default_factory=list)

    @property
    def level(self) -> str:
        return worst([ln.level for ln in self.lines])


def worst(levels: list[str]) -> str:
    """The most severe level; n/a only if nothing was scored."""
    return max(levels, key=_RANK.__getitem__, default=NA)


def _isnan(v: float | str) -> bool:
    return isinstance(v, float) and np.isnan(v)


def band_max(v: float, bounds: tuple[float, float], strict: bool = False) -> str:
    """Lower is better: pass if v <= bounds[0] (< if strict), warn if <= bounds[1] (< if strict)."""
    if _isnan(v):
        return NA
    v = round(float(v), 6)  # float32 targets: 0.79999995 must not land on the other side of 0.80
    ok = (lambda a, b: a < b) if strict else (lambda a, b: a <= b)
    return PASS if ok(v, bounds[0]) else WARN if ok(v, bounds[1]) else FAIL


def band_min(v: float, bounds: tuple[float, float]) -> str:
    """Higher is better: pass if v >= bounds[0], warn if v >= bounds[1]."""
    if _isnan(v):
        return NA
    v = round(float(v), 6)
    return PASS if v >= bounds[0] else WARN if v >= bounds[1] else FAIL


def _warn_if(cond: bool, v: float) -> str:
    return NA if _isnan(v) else (WARN if cond else PASS)


def grade(m: dict) -> list[CheckResult]:
    """Grade the metrics of ``quality.all_metrics`` into the six checks."""
    f = lambda k: float(m[k])  # noqa: E731
    out: list[CheckResult] = []

    out.append(CheckResult(1, TITLES[1], [
        Line("calibration extent x", f("calib_extent_x"), band_min(f("calib_extent_x"), T["extent_x"])),
        Line("calibration extent y", f("calib_extent_y"), band_min(f("calib_extent_y"), T["extent_y"])),
        Line("test targets outside calibration hull", f("test_outside_hull"),
             band_max(f("test_outside_hull"), T["outside_hull"])),
    ]))
    out.append(CheckResult(2, TITLES[2], [
        Line("valid-frame share", f("valid_fraction"), band_min(f("valid_fraction"), T["valid_fraction"])),
        Line("targets with < 10 valid frames", f("thin_targets"), band_max(f("thin_targets"), T["thin_targets"])),
        Line("dropped-frame gap share", f("dropped_gap_share"),
             band_max(f("dropped_gap_share"), T["dropped_gap_share"])),
    ]))
    out.append(CheckResult(3, TITLES[3], [
        Line("calibration -> test targets, mean deg", f("heldout_mean"),
             band_max(f("heldout_mean"), T["heldout_mean"]), "within-session, not cross-session"),
        Line("leave-one-target-out, mean deg", f("loto_mean"), band_max(f("loto_mean"), T["loto_mean"]),
             "within-session; edge dots are extrapolated in a 9-dot session"),
        Line("share of targets worse than 8 deg", f("loto_far_share"),
             band_max(f("loto_far_share"), T["far_share"], strict=True),
             "in a 9-dot session one bad dot = 0.11: look at that dot, not a verdict on the session"),
    ]))
    out.append(CheckResult(4, TITLES[4], [
        Line("compression slope x", f("slope_x"), _warn_if(f("slope_x") < T["slope_x_warn"], f("slope_x")),
             "1.0 = none"),
        Line("compression slope y", f("slope_y"), _warn_if(f("slope_y") < T["slope_y_warn"], f("slope_y")),
             "y is noisier and compressed more"),
        Line("within-fixation spread (jitter), deg", f("spread_deg"),
             _warn_if(f("spread_deg") > T["spread_warn"], f("spread_deg"))),
        Line("bias, deg", f("bias_deg"), NA if _isnan(f("bias_deg")) else INFO, "reported"),
        Line("bias share of squared error", f("bias_share"), NA if _isnan(f("bias_share")) else INFO,
             "high = needs better features / calibration, not smoothing"),
    ]))
    out.append(CheckResult(5, TITLES[5], [
        Line("head shift first vs last quarter, cm", f("head_shift_cm"),
             band_max(f("head_shift_cm"), T["head_shift_cm"])),
        Line("head rotation first vs last quarter, deg", f("head_rot_deg"),
             band_max(f("head_rot_deg"), T["head_rot_deg"])),
        Line("late minus early error, deg", f("late_minus_early_deg"),
             NA if _isnan(f("late_minus_early_deg")) else INFO, "printed, not scored"),
    ]))
    tags_present, tags_expected = f("tags_present"), f("tags_expected")
    tags_level = str(m.get("tags_verdict") or (PASS if tags_present >= tags_expected else WARN))
    tags_note = str(m.get("tags_issues") or "") or "person_id missing fails a new session, other gaps warn"
    out.append(CheckResult(6, TITLES[6], [
        Line("face brightness (0-255)", f("face_brightness"), band_min(f("face_brightness"), T["face_brightness"]),
             "existing gate: 60 minimum, 100 recommended"),
        Line("lighting label", str(m["lighting_label"]) or "(none)", NA),
        Line("condition tags present", f"{int(tags_present)}/{int(tags_expected)}", tags_level, tags_note),
    ]))
    return out


def overall(results: list[CheckResult]) -> str:
    """Worst level over all checks."""
    return worst([r.level for r in results])


def format_report(name: str, results: list[CheckResult]) -> str:
    """Plain-text report: one line per check, then the detail lines under it."""
    def val(v: float | str) -> str:
        return v if isinstance(v, str) else ("nan" if np.isnan(v) else f"{v:.3g}")
    rows = [f"Session quality report: {name}", f"OVERALL: {overall(results).upper()}", ""]
    for r in results:
        rows.append(f"[{r.level.upper():4}] {r.check}. {r.title}")
        for ln in r.lines:
            tail = f"  ({ln.note})" if ln.note else ""
            rows.append(f"         {ln.level:4}  {ln.name}: {val(ln.value)}{tail}")
    return "\n".join(rows)
