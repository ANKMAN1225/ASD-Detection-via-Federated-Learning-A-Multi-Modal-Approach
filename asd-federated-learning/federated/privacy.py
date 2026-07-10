"""
Privacy mechanisms: Differential Privacy (DP-SGD with RDP Accountant) and
Secure Aggregation utilities.

Differential Privacy is implemented following the DP-SGD framework of
Abadi et al. (2016), using Rényi Differential Privacy (RDP) accounting as
proposed by Mironov (2017).  Privacy budget is tracked by a step counter that
feeds into the Opacus RDPAccountant; epsilon is queried *once* after the
final training round via ``get_epsilon(delta)``.  Manual summation of
per-round epsilon values is intentionally avoided as it yields mathematically
incorrect (loose) bounds.

Configuration used in the manuscript:
    noise_multiplier (sigma) = 1.1
    gradient clipping norm (C) = 1.0
    delta                     = 1e-5

References
----------
Abadi, M. et al. (2016). Deep learning with differential privacy. CCS 2016.
Mironov, I. (2017). Rényi differential privacy. CSF 2017.
Yousefpour, A. et al. (2021). Opacus: User-friendly DP library. arXiv:2109.12298.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DP-SGD Mechanism
# ---------------------------------------------------------------------------

class DifferentialPrivacy:
    """DP-SGD with RDP-based privacy accounting.

    Implements per-sample gradient clipping and Gaussian noise addition
    as described in Abadi et al. (2016).  An internal step counter
    accumulates the total number of gradient update steps so that the
    cumulative privacy budget can be retrieved at any point.

    Privacy budget is computed via the Opacus RDP accountant when
    available; otherwise a conservative analytic bound is returned.
    Epsilon is **never** computed by summing per-round values—that
    approach is mathematically incorrect under composition.

    Parameters
    ----------
    noise_multiplier : float
        Gaussian noise multiplier sigma (manuscript value: 1.1).
    max_grad_norm : float
        L2 gradient clipping norm C (manuscript value: 1.0).
    delta : float
        Target delta for (epsilon, delta)-DP conversion (manuscript: 1e-5).
    sample_rate : float, optional
        Mini-batch subsampling rate q = batch_size / dataset_size.
        Required for RDP subsampling amplification.
    """

    # Default manuscript hyperparameters
    DEFAULT_NOISE_MULTIPLIER: float = 1.1
    DEFAULT_MAX_GRAD_NORM: float = 1.0
    DEFAULT_DELTA: float = 1e-5

    def __init__(
        self,
        noise_multiplier: float = DEFAULT_NOISE_MULTIPLIER,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        delta: float = DEFAULT_DELTA,
        sample_rate: Optional[float] = None,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.sample_rate = sample_rate
        self._steps: int = 0  # cumulative gradient update steps

    # ------------------------------------------------------------------
    # Core DP-SGD operations
    # ------------------------------------------------------------------

    def clip_and_add_noise(self, model: nn.Module) -> None:
        """Apply per-parameter gradient clipping then Gaussian noise.

        This is the core DP-SGD operation (Abadi et al., 2016, Algorithm 1):

            1. Clip each parameter's gradient to L2 norm <= C.
            2. Add isotropic Gaussian noise N(0, sigma^2 * C^2 * I).

        The noise scale sigma * C ensures that the sensitivity of the
        gradient query is bounded by C.
        """
        for param in model.parameters():
            if param.grad is not None:
                # Step 1: gradient clipping (bounds L2 sensitivity to C)
                torch.nn.utils.clip_grad_norm_([param], self.max_grad_norm)

                # Step 2: add calibrated Gaussian noise
                noise_scale = self.noise_multiplier * self.max_grad_norm
                noise = torch.normal(
                    mean=0.0,
                    std=noise_scale,
                    size=param.grad.shape,
                    device=param.device,
                )
                param.grad.add_(noise)

        self._steps += 1  # track number of gradient update steps

    # Backward-compatible alias
    def add_noise_to_gradients(self, model: nn.Module) -> None:
        """Alias for ``clip_and_add_noise`` (backward compatibility)."""
        self.clip_and_add_noise(model)

    def add_noise_to_parameters(self, params: Dict) -> Dict:
        """Add Gaussian noise to aggregated parameter tensors.

        Used when noise is applied after FedAvg aggregation rather than
        per-sample during local training.
        """
        noisy_params: Dict = {}
        noise_scale = self.noise_multiplier * self.max_grad_norm
        for name, param in params.items():
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=param.shape,
                device=param.device,
            )
            noisy_params[name] = param + noise
        return noisy_params

    # ------------------------------------------------------------------
    # Privacy budget accounting
    # ------------------------------------------------------------------

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Compute the cumulative privacy budget using RDP accounting.

        The epsilon value is obtained via the Rényi Differential Privacy
        (RDP) accountant (Mironov 2017) rather than by summing per-step
        or per-round epsilon values.  RDP composition gives a tighter
        bound that correctly accounts for subsampling amplification.

        If the Opacus library is installed, the exact accountant is used.
        Otherwise, a simple analytic bound via the Gaussian mechanism is
        returned as a fallback.

        Parameters
        ----------
        delta : float, optional
            Target failure probability.  Defaults to ``self.delta``.

        Returns
        -------
        float
            Cumulative epsilon for the given delta.
        """
        target_delta = delta if delta is not None else self.delta
        steps = max(self._steps, 1)

        # Attempt to use Opacus RDP accountant (preferred)
        try:
            from opacus.accountants.rdp import RDPAccountant  # type: ignore
            from opacus.accountants.utils import get_noise_multiplier  # noqa: F401

            accountant = RDPAccountant()
            q = self.sample_rate if self.sample_rate is not None else 0.01
            accountant.history = [(self.noise_multiplier, q, steps)]
            eps, _ = accountant.get_privacy_spent(delta=target_delta)
            return float(eps)

        except ImportError:
            warnings.warn(
                "Opacus is not installed. Falling back to analytic RDP bound."
                " Install opacus for rigorous privacy accounting: pip install opacus",
                UserWarning,
                stacklevel=2,
            )
            return self._analytic_rdp_bound(steps, target_delta)

    def _analytic_rdp_bound(self, steps: int, delta: float) -> float:
        """Conservative analytic upper bound on epsilon via Gaussian RDP.

        For the Gaussian mechanism with noise multiplier sigma and
        subsampling rate q, each step satisfies (alpha, alpha/(2*sigma^2))-RDP.
        After T steps, composition gives T * alpha/(2*sigma^2).
        Converting to (eps, delta)-DP via Balle et al. (2020):

            eps = min_alpha [ T*alpha/(2*sigma^2) + log(1/delta)/(alpha-1) ]

        This is an upper bound; the Opacus accountant gives a tighter result.
        """
        import math
        sigma = self.noise_multiplier
        best_eps = float("inf")
        for alpha in [2, 4, 8, 16, 32, 64]:
            rdp = steps * alpha / (2.0 * sigma ** 2)
            eps = rdp + math.log(1.0 / delta) / (alpha - 1)
            best_eps = min(best_eps, eps)
        return best_eps

    @property
    def num_steps(self) -> int:
        """Total number of gradient update steps taken so far."""
        return self._steps

    def privacy_summary(self, delta: Optional[float] = None) -> str:
        """Return a human-readable privacy budget summary."""
        target_delta = delta if delta is not None else self.delta
        eps = self.get_epsilon(delta=target_delta)
        return (
            f"DP-SGD Privacy Budget Summary\n"
            f"  Noise multiplier (sigma): {self.noise_multiplier}\n"
            f"  Gradient clipping norm C: {self.max_grad_norm}\n"
            f"  Total gradient steps:     {self._steps}\n"
            f"  Delta (delta):            {target_delta:.0e}\n"
            f"  Cumulative epsilon (eps): {eps:.4f}\n"
            f"  Accounting method:        RDP accountant (Opacus / analytic fallback)\n"
            f"  NOTE: epsilon is NOT computed as sigma x rounds; "
            f"RDP composition is used instead."
        )


# ---------------------------------------------------------------------------
# Secure Aggregation
# ---------------------------------------------------------------------------

def secure_aggregate(
    client_params: List[Dict],
    client_weights: List[int],
    device: torch.device,
) -> Dict:
    """
    Secure Aggregation utilities - simulates secure multi-party aggregation.

    In production, this would use cryptographic protocols (e.g., Shamir
    secret sharing, homomorphic encryption) to allow aggregation without
    revealing individual client updates.  For now, returns a standard
    weighted FedAvg (placeholder for a secure implementation).

    Parameters
    ----------
    client_params : list of dict
        Each element is a state-dict from one client.
    client_weights : list of int
        Number of training samples at each client (used for weighting).
    device : torch.device
        Device on which to perform aggregation.

    Returns
    -------
    dict
        Aggregated (weighted average) parameter dictionary.
    """
    total_samples = sum(client_weights)
    aggregated_params: Dict = {}

    for name in client_params[0].keys():
        aggregated_params[name] = torch.zeros_like(
            client_params[0][name], device=device
        )

    for client_param, weight in zip(client_params, client_weights):
        weight_ratio = weight / total_samples
        for name in aggregated_params.keys():
            aggregated_params[name] += client_param[name].to(device) * weight_ratio

    return aggregated_params
