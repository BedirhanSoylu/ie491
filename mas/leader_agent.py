"""
Leader Agent — Particle Filter + Confidence Interval decision trigger.

Decision logic (evaluated each channel):
  1. REPLACE  — if PF critical_probability > 90 % (highest priority)
  2. TAKE_IMAGE — if observed wear signal falls outside the CI that was
                  predicted for this channel before it arrived (model mismatch)
  3. CONTINUE  — signal is within CI and risk is below threshold

After TAKE_IMAGE the caller should call incorporate_image_truth() with the
K-Means label so that the particle cloud is collapsed to the ground truth.
"""
from __future__ import annotations

from mas.agents.particle_filter import ParticleFilter

CRITICAL_PROB_THRESHOLD = 0.90

# Canonical wear level per K-Means label (centre of the cluster in [0,1])
_WEAR_LEVEL_MAP: dict[str, float] = {
    "FACTORY_NEW": 0.10,
    "MID_WORN":    0.45,
    "CRITICAL":    0.80,
}


class LeaderAgent:
    def __init__(self, pf: ParticleFilter) -> None:
        self.pf = pf
        # Pre-compute CI for the first incoming observation
        mean, lo, hi = self.pf.predict_next_ci()
        self._next_ci_mean: float = mean
        self._next_ci_low:  float = lo
        self._next_ci_high: float = hi

    # ------------------------------------------------------------------
    # Main decision interface
    # ------------------------------------------------------------------

    def decide(self, observed_signal: float) -> str:
        """
        Parameters
        ----------
        observed_signal : float in [0, 1]
            Normalised wear signal from the Force Agent for the current channel.

        Returns
        -------
        "CONTINUE" | "TAKE_IMAGE" | "REPLACE"
        """
        # Run full PF update with this observation
        self.pf.step(observed_signal)

        # Priority 1: replace if most particles are in critical zone
        if self.pf.critical_probability() > CRITICAL_PROB_THRESHOLD:
            self._refresh_next_ci()
            return "REPLACE"

        # Priority 2: model mismatch → image needed
        if observed_signal < self._next_ci_low or observed_signal > self._next_ci_high:
            decision = "TAKE_IMAGE"
        else:
            decision = "CONTINUE"

        self._refresh_next_ci()
        return decision

    def incorporate_image_truth(self, kmeans_label: str) -> None:
        """
        Collapse the PF around the K-Means ground-truth wear level and
        refresh the CI prediction.
        """
        wear_level = _WEAR_LEVEL_MAP.get(kmeans_label, 0.45)
        self.pf.collapse_to_state(wear_level)
        self._refresh_next_ci()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def current_ci(self) -> tuple[float, float]:
        """CI that will be used to evaluate the NEXT incoming observation."""
        return self._next_ci_low, self._next_ci_high

    def current_ci_mean(self) -> float:
        return self._next_ci_mean

    def rul(self) -> int:
        """Estimated remaining channels until critical (from PF simulation)."""
        return self.pf.rul_estimate()

    # ------------------------------------------------------------------

    def _refresh_next_ci(self) -> None:
        mean, lo, hi = self.pf.predict_next_ci()
        self._next_ci_mean = mean
        self._next_ci_low  = lo
        self._next_ci_high = hi
