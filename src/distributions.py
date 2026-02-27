"""Custom distributions for two-stage frozen-mu LightGBMLSS models."""

from __future__ import annotations

import numpy as np
from lightgbmlss.distributions.Gaussian import Gaussian
from lightgbmlss.distributions.distribution_utils import DistributionClass


def bounded_sigmoid_fn(x, min_val=0.1, max_val=0.6):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x_clamped = torch.clamp(x, -20, 20)
            sigmoid = 1.0 / (1.0 + torch.exp(-x_clamped))
            return min_val + (max_val - min_val) * sigmoid
    except ImportError:
        pass

    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return min_val + (max_val - min_val) * sigmoid


class BoundedSigmoidFn:
    def __init__(self, min_val=0.15, max_val=0.45):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        return bounded_sigmoid_fn(x, self.min_val, self.max_val)

    def __reduce__(self):
        return (BoundedSigmoidFn, (self.min_val, self.max_val))


class GaussianFrozenLoc(DistributionClass):
    _class_printed = False

    def __init__(self, stabilization="MAD"):
        from lightgbmlss.distributions.Gaussian import Gaussian_Torch, exp_fn, identity_fn

        distribution = Gaussian_Torch
        param_dict = {"loc": identity_fn, "scale": exp_fn}

        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn="nll",
        )

        self.dist_class = Gaussian()

    def compute_gradients_and_hessians(self, loss, predt, weights=None):
        if not GaussianFrozenLoc._class_printed:
            print("*** Using GaussianFrozenLoc (frozen mu, learning sigma only) ***")
            GaussianFrozenLoc._class_printed = True

        grad, hess = self.dist_class.compute_gradients_and_hessians(loss, predt, weights)
        if grad.ndim == 1 and self.n_dist_param == 2:
            n_samples = len(grad) // 2
            grad[:n_samples] = 0.0
            hess[:n_samples] = 1e-12
        return grad, hess


class GaussianFrozenLocBounded(DistributionClass):
    _class_printed = False

    def __init__(self, stabilization="MAD", sigma_min=0.15, sigma_max=0.45):
        from lightgbmlss.distributions.Gaussian import Gaussian_Torch, identity_fn

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        bounded_scale_fn = BoundedSigmoidFn(sigma_min, sigma_max)

        distribution = Gaussian_Torch
        param_dict = {"loc": identity_fn, "scale": bounded_scale_fn}

        super().__init__(
            distribution=distribution,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn="nll",
        )

    def compute_gradients_and_hessians(self, loss, predt, weights=None):
        if not GaussianFrozenLocBounded._class_printed:
            print(
                f"*** Using GaussianFrozenLocBounded (frozen mu, sigma in [{self.sigma_min}, {self.sigma_max}]) ***"
            )
            GaussianFrozenLocBounded._class_printed = True

        grad, hess = super().compute_gradients_and_hessians(loss, predt, weights)
        if grad.ndim == 1 and self.n_dist_param == 2:
            n_samples = len(grad) // 2
            grad[:n_samples] = 0.0
            hess[:n_samples] = 1e-12
        return grad, hess
