"""
Particle Filter for tool wear RUL estimation.

State: wear_level in [0, 1]
  0 = factory new, 1 = fully worn (critical threshold = 0.75)

Motion model (irreversible wear increase):
  wear[t+1] = wear[t] + |N(mu_drift, sigma_drift)|

Observation model:
  p(obs | wear) ~ exp(-0.5 * ((wear - obs) / sigma_obs)^2)
"""
from __future__ import annotations

import numpy as np

CRITICAL_THRESHOLD = 0.75
N_PARTICLES = 500


class ParticleFilter:
    def __init__(
        self,
        n_particles: int = N_PARTICLES,
        critical_threshold: float = CRITICAL_THRESHOLD,
        mu_drift: float = 0.018,
        sigma_drift: float = 0.008,
        sigma_obs: float = 0.07,
    ) -> None:
        self.n = n_particles
        self.critical_threshold = critical_threshold
        self.mu_drift = mu_drift
        self.sigma_drift = sigma_drift
        self.sigma_obs = sigma_obs

        # Initialise near factory-new state
        self.particles = np.clip(np.random.normal(0.05, 0.03, self.n), 0.0, 1.0)
        self.weights = np.ones(self.n) / self.n

    # ------------------------------------------------------------------
    # Core PF cycle
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """Propagate particles forward one step (wear is irreversible)."""
        drift = np.abs(np.random.normal(self.mu_drift, self.sigma_drift, self.n))
        self.particles = np.clip(self.particles + drift, 0.0, 1.0)

    def update(self, observation: float) -> None:
        """Weight particles by Gaussian likelihood of the observed wear signal."""
        log_w = -0.5 * ((self.particles - observation) / self.sigma_obs) ** 2
        log_w -= log_w.max()           # numerical stability
        w = np.exp(log_w) + 1e-300
        self.weights = w / w.sum()

    def resample(self) -> None:
        """Systematic resampling to avoid weight degeneracy."""
        positions = (np.arange(self.n) + np.random.uniform()) / self.n
        cumsum = np.cumsum(self.weights)
        idx = np.searchsorted(cumsum, positions)
        self.particles = self.particles[idx]
        self.weights = np.ones(self.n) / self.n

    def step(self, observation: float) -> None:
        """Full update cycle: predict -> update -> resample."""
        self.predict()
        self.update(observation)
        self.resample()

    # ------------------------------------------------------------------
    # State queries (all read-only — do not modify self.particles)
    # ------------------------------------------------------------------

    def state_mean(self) -> float:
        return float(np.average(self.particles, weights=self.weights))

    def state_ci(self, alpha: float = 0.90) -> tuple[float, float]:
        """Confidence interval of current state distribution."""
        lo = (1.0 - alpha) / 2.0 * 100
        hi = (1.0 + alpha) / 2.0 * 100
        return float(np.percentile(self.particles, lo)), float(np.percentile(self.particles, hi))

    def predict_next_ci(self, alpha: float = 0.90) -> tuple[float, float, float]:
        """
        Predict CI for the NEXT observation without committing the step.
        Returns (mean, ci_low, ci_high).
        """
        temp = self.particles.copy()
        drift = np.abs(np.random.normal(self.mu_drift, self.sigma_drift, self.n))
        temp = np.clip(temp + drift, 0.0, 1.0)
        lo = (1.0 - alpha) / 2.0 * 100
        hi = (1.0 + alpha) / 2.0 * 100
        return float(np.mean(temp)), float(np.percentile(temp, lo)), float(np.percentile(temp, hi))

    def critical_probability(self) -> float:
        """Fraction of particles at or above the critical threshold."""
        return float(np.mean(self.particles >= self.critical_threshold))

    def rul_estimate(self, n_steps: int = 60) -> int:
        """
        Simulate forward and return the number of steps until >50 % of
        particles exceed the critical threshold.  Capped at n_steps.
        """
        temp = self.particles.copy()
        for step in range(1, n_steps + 1):
            drift = np.abs(np.random.normal(self.mu_drift, self.sigma_drift, self.n))
            temp = np.clip(temp + drift, 0.0, 1.0)
            if np.mean(temp >= self.critical_threshold) >= 0.5:
                return step
        return n_steps

    def future_trajectory(
        self, n_steps: int = 15
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate n_steps ahead from current state.
        Returns (means, ci_lows, ci_highs) each of length n_steps.
        Does NOT modify self.particles.
        """
        temp = self.particles.copy()
        means, lows, highs = [], [], []
        for _ in range(n_steps):
            drift = np.abs(np.random.normal(self.mu_drift, self.sigma_drift, self.n))
            temp = np.clip(temp + drift, 0.0, 1.0)
            means.append(float(np.mean(temp)))
            lows.append(float(np.percentile(temp, 5.0)))
            highs.append(float(np.percentile(temp, 95.0)))
        return np.array(means), np.array(lows), np.array(highs)

    def particle_snapshot(self, max_display: int = 120) -> np.ndarray:
        """Return a random subset of current particles for visualisation."""
        idx = np.random.choice(self.n, min(max_display, self.n), replace=False)
        return self.particles[idx].copy()

    # ------------------------------------------------------------------
    # Ground-truth reset (called after K-Means classification on image)
    # ------------------------------------------------------------------

    def collapse_to_state(self, wear_level: float, sigma: float = 0.025) -> None:
        """
        Collapse particle cloud tightly around a known wear level.
        Called after Image Agent returns a K-Means ground-truth label.
        """
        self.particles = np.clip(
            np.random.normal(wear_level, sigma, self.n), 0.0, 1.0
        )
        self.weights = np.ones(self.n) / self.n
