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
TAKE_IMAGE_COOLDOWN     = 7    # minimum channels between consecutive TAKE_IMAGE decisions
_REPLACE_HYSTERESIS     = 3    # consecutive critical steps required before REPLACE

# Canonical wear level per K-Means label (centre of the cluster in [0,1])
_WEAR_LEVEL_MAP: dict[str, float] = {
    "FACTORY_NEW": 0.08,
    "MID_WORN":    0.20,
    "CRITICAL":    0.38,
}


class LeaderAgent:
    def __init__(self, pf: ParticleFilter) -> None:
        self.pf = pf
        # Pre-compute CI for the first incoming observation
        mean, lo, hi = self.pf.predict_next_ci()
        self._next_ci_mean: float = mean
        self._next_ci_low:  float = lo
        self._next_ci_high: float = hi
        # Throttle image-take decisions
        self._steps_since_take_image: int = TAKE_IMAGE_COOLDOWN  # allow on first channel
        self._consecutive_critical:   int = 0

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
        self.pf.step(observed_signal)
        self._steps_since_take_image += 1

        # Track consecutive critical steps for REPLACE hysteresis
        if self.pf.critical_probability() > CRITICAL_PROB_THRESHOLD:
            self._consecutive_critical += 1
        else:
            self._consecutive_critical = 0

        if self._consecutive_critical >= _REPLACE_HYSTERESIS:
            self._refresh_next_ci()
            return "REPLACE"

        # TAKE_IMAGE only when signal outside CI AND cooldown elapsed
        outside_ci = (
            observed_signal < self._next_ci_low
            or observed_signal > self._next_ci_high
        )
        if outside_ci and self._steps_since_take_image >= TAKE_IMAGE_COOLDOWN:
            decision = "TAKE_IMAGE"
            self._steps_since_take_image = 0
        else:
            decision = "CONTINUE"

        self._refresh_next_ci()
        return decision

    def decide_fused(
        self,
        wear_signal: float,
        ft_norm: float,
        fn_norm: float,
    ) -> str:
        """PF step with fused observation (wear + normalised Ft + Fn)."""
        self.pf.step_fused(wear_signal, ft_norm, fn_norm)
        self._steps_since_take_image += 1

        if self.pf.critical_probability() > CRITICAL_PROB_THRESHOLD:
            self._consecutive_critical += 1
        else:
            self._consecutive_critical = 0

        if self._consecutive_critical >= _REPLACE_HYSTERESIS:
            self._refresh_next_ci()
            return "REPLACE"

        outside_ci = (
            wear_signal < self._next_ci_low
            or wear_signal > self._next_ci_high
        )
        if outside_ci and self._steps_since_take_image >= TAKE_IMAGE_COOLDOWN:
            decision = "TAKE_IMAGE"
            self._steps_since_take_image = 0
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
