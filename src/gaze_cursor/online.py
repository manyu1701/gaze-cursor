"""Self-recalibration from clicks (M3 step 6): every click is a labelled gaze sample (you usually look
where you click), so the gaze model can keep adjusting during normal use without any extra input.

- ``OnlineRidge``: the quick calibration's ridge model (standardize -> ridge, unpenalized intercept),
  re-expressed so it can be updated one sample at a time by recursive least squares with forgetting.
  It starts **exactly** at the batch fit; the calibration counts as ``prior_samples`` samples (one per
  dot: the unit of data is a target, not a frame).
- ``click_label``: the gaze features of the fixation that guided a click (median over a window before
  it, before the hand moved).
- ``prequential``: replay a recorded Cursor session click by click: predict each click with the frozen
  and the online model *before* updating on it (honest, out-of-sample at every step).
- ``Cusum``: a two-sided CUSUM on residuals, a drift diagnostic (reported only in this round).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LABEL_WINDOW_MS = (-400.0, -100.0)
PRIOR_SAMPLES = 9.0        # the calibration's weight: its 9 dots
FORGET = 0.98              # per accepted click: memory ~ 1 / (1 - 0.98) = 50 clicks
GATE_DEG = 6.0             # update only on clicks this close to the current prediction
EVAL_GATE_DEG = 10.0       # evaluate on clicks this close to the FROZEN prediction (model-independent)


@dataclass
class OnlineRidge:
    mean: np.ndarray           # standardizer (from the calibration frames)
    scale: np.ndarray
    theta: np.ndarray          # (n_features + 1, 2): coefficients in standardized space, intercept last
    P: np.ndarray              # (n_features + 1, n_features + 1) inverse information
    forget: float = FORGET
    n_updates: int = 0

    @classmethod
    def from_calibration(cls, x: np.ndarray, y: np.ndarray, alpha: float, prior_samples: float = PRIOR_SAMPLES,
                         forget: float = FORGET) -> OnlineRidge:
        """Batch ridge on the calibration frames, identical to ``GazeModel('ridge', alpha)``."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 0, scale, 1.0)
        z = (x - mean) / scale
        n, k = z.shape
        a = np.zeros((k + 1, k + 1))
        a[:k, :k] = z.T @ z + alpha * np.eye(k)
        a[k, k] = n                                   # z is centred, so the intercept block decouples
        b = np.vstack([z.T @ y, y.sum(axis=0)])
        theta = np.linalg.solve(a, b)
        p = np.linalg.inv(a) * (n / prior_samples)    # the calibration is worth prior_samples samples
        return cls(mean, scale, theta, p, forget)

    def _phi(self, x: np.ndarray) -> np.ndarray:
        z = (np.atleast_2d(np.asarray(x, float)) - self.mean) / self.scale
        return np.hstack([z, np.ones((len(z), 1))])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._phi(x) @ self.theta

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """One RLS step with forgetting on a single sample (features ``x``, screen point ``y``)."""
        phi = self._phi(x)[0]
        pphi = self.P @ phi
        gain = pphi / (self.forget + phi @ pphi)
        self.theta = self.theta + np.outer(gain, np.asarray(y, float) - phi @ self.theta)
        self.P = (self.P - np.outer(gain, pphi)) / self.forget
        self.n_updates += 1

    def copy(self) -> OnlineRidge:
        return OnlineRidge(self.mean.copy(), self.scale.copy(), self.theta.copy(), self.P.copy(), self.forget,
                           self.n_updates)


def click_label(t_frames: np.ndarray, features: np.ndarray, t_click: float,
                window: tuple[float, float] = LABEL_WINDOW_MS, min_frames: int = 3) -> np.ndarray | None:
    """Median features over ``t_click + window`` (rows with NaN = no face / blink are skipped); None if
    fewer than ``min_frames`` usable frames."""
    sel = (t_frames >= t_click + window[0]) & (t_frames <= t_click + window[1])
    rows = features[sel]
    rows = rows[~np.isnan(rows).any(axis=1)]
    return np.median(rows, axis=0) if len(rows) >= min_frames else None


@dataclass
class Cusum:
    """Two-sided CUSUM on a residual stream (deg): flags a sustained shift above ``k`` (slack); fires when
    the running sum exceeds ``h``, then resets."""
    k: float = 1.0
    h: float = 6.0
    hi: float = 0.0
    lo: float = 0.0
    alarms: list[int] = field(default_factory=list)

    def update(self, r: float, i: int) -> bool:
        self.hi = max(0.0, self.hi + r - self.k)
        self.lo = max(0.0, self.lo - r - self.k)
        if self.hi > self.h or self.lo > self.h:
            self.alarms.append(i)
            self.hi = self.lo = 0.0
            return True
        return False


@dataclass
class ClickResult:
    t_ms: float
    click: np.ndarray
    frozen: np.ndarray         # prediction of the frozen calibration model
    online: np.ndarray         # prediction of the online model BEFORE updating on this click
    err_frozen_deg: float
    err_online_deg: float
    updated: bool
    evaluated: bool            # within EVAL_GATE_DEG of the frozen prediction
    segment: int


def prequential(t_frames: np.ndarray, features: np.ndarray, clicks: list[tuple[float, float, float, int]],
                priors: dict[int, OnlineRidge], px_per_deg: float, gate_deg: float = GATE_DEG,
                eval_gate_deg: float = EVAL_GATE_DEG) -> list[ClickResult]:
    """Replay clicks in time order. ``clicks``: (t_ms, x, y, segment); ``priors``: the calibration model
    per segment (a recalibration starts a new segment from its own fresh prior). Each click is predicted
    by both models before the online one may learn from it."""
    out: list[ClickResult] = []
    online: dict[int, OnlineRidge] = {}
    for t, cx, cy, seg in sorted(clicks):
        if seg not in priors:
            continue
        x = click_label(t_frames, features, t)
        if x is None:
            continue
        frozen = priors[seg]
        model = online.setdefault(seg, frozen.copy())
        target = np.array([cx, cy])
        pf, po = frozen.predict(x)[0], model.predict(x)[0]
        ef = float(np.hypot(*(pf - target)) / px_per_deg)
        eo = float(np.hypot(*(po - target)) / px_per_deg)
        update = eo <= gate_deg
        if update:
            model.update(x, target)
        out.append(ClickResult(t, target, pf, po, ef, eo, update, ef <= eval_gate_deg, seg))
    return out
