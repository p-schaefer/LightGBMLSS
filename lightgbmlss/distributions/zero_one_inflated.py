import torch
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus

from torch.distributions import NegativeBinomial, Poisson, Gamma, LogNormal, Beta
from pyro.distributions import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.distributions.util import is_identically_one, is_identically_zero

class ZeroOneInflatedDistribution(TorchDistribution):
    """
    Generic Zero and One Inflated distribution.

    This distribution assigns extra probability mass at the boundaries 0 and 1,
    while using a continuous base distribution (e.g. Beta) for values in (0, 1).

    Parameters
    ----------
    base_dist : torch.distributions.Distribution
        The continuous distribution (Beta in our case).
    gate_0 : torch.Tensor or None
        Probability of extra zeros (must lie in [0,1]).
    gate_0_logits : torch.Tensor or None
        Logits for the probability of extra zeros.
    gate_1 : torch.Tensor or None
        Probability of extra ones.
    gate_1_logits : torch.Tensor or None
        Logits for the probability of extra ones.
    validate_args : bool, optional
        Whether to validate input arguments.
    """
    arg_constraints = {
        "gate_0": constraints.unit_interval,
        "gate_0_logits": constraints.real,
        "gate_1": constraints.unit_interval,
        "gate_1_logits": constraints.real,
    }

    def __init__(self, base_dist, *, 
                 gate_0=None, gate_0_logits=None, 
                 gate_1=None, gate_1_logits=None, 
                 validate_args=None):
        # Exactly one of (gate_0, gate_0_logits) must be provided:
        if (gate_0 is None) == (gate_0_logits is None):
            raise ValueError("Either `gate_0` or `gate_0_logits` must be specified, but not both.")
        if (gate_1 is None) == (gate_1_logits is None):
            raise ValueError("Either `gate_1` or `gate_1_logits` must be specified, but not both.")

        # Determine batch shape from the provided gates and base distribution.
        if gate_0 is not None and gate_1 is not None:
            batch_shape = broadcast_shape(gate_0.shape, gate_1.shape, base_dist.batch_shape)
            self._gate_0, self._gate_1 = broadcast_all(gate_0, gate_1)
        else:
            # If using logits, they should be provided for both.
            batch_shape = broadcast_shape(gate_0_logits.shape, gate_1_logits.shape, base_dist.batch_shape)
            self._gate_0 = logits_to_probs(gate_0_logits.expand(batch_shape))
            self._gate_1 = logits_to_probs(gate_1_logits.expand(batch_shape))

        # Check that the combined gate probability does not exceed 1.
        if (self._gate_0 + self._gate_1 > 1).any():
            raise ValueError("The sum of gate_0 and gate_1 must not exceed 1.")

        self.base_dist = base_dist.expand(batch_shape)
        self._batch_shape = batch_shape
        event_shape = torch.Size()
        super().__init__(batch_shape, event_shape, validate_args)

    @lazy_property
    def gate_0(self):
        return self._gate_0

    @lazy_property
    def gate_1(self):
        return self._gate_1

    @lazy_property
    def beta_weight(self):
        # The weight for the continuous (Beta) component.
        return 1 - self.gate_0 - self.gate_1

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Avoid numerical issues near the boundaries:
        epsilon = abs(torch.finfo(value.dtype).eps)
        value_cont = value.clamp(epsilon, 1 - epsilon)

        # Create masks for points exactly equal to zero or one.
        is_zero = (value == 0)
        is_one = (value == 1)
        is_cont = ~(is_zero | is_one)

        # Log probability for the continuous region.
        log_prob_cont = torch.log(self.beta_weight + epsilon) + self.base_dist.log_prob(value_cont)
        # Log probabilities for the boundaries.
        log_prob_zero = torch.log(self.gate_0 + epsilon)
        log_prob_one = torch.log(self.gate_1 + epsilon)

        # Select according to the value.
        logp = torch.where(is_zero, log_prob_zero,
                           torch.where(is_one, log_prob_one, log_prob_cont))
        return logp

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            u = torch.rand(shape, device=self.base_dist.concentration1.device)
            # Determine thresholds:
            threshold_zero = self.gate_0.expand(shape)
            threshold_one = self.gate_0.expand(shape) + self.gate_1.expand(shape)
            # Sample continuous component.
            cont_sample = self.base_dist.expand(shape).sample()
            sample = torch.where(u < threshold_zero, torch.zeros_like(u),
                                 torch.where(u < threshold_one, torch.ones_like(u), cont_sample))
        return sample

    @lazy_property
    def mean(self):
        return (self.beta_weight * self.base_dist.mean)

    @lazy_property
    def variance(self):
        return (self.beta_weight * (self.base_dist.variance + self.base_dist.mean**2)) - self.mean**2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        gate_0 = self.gate_0.expand(batch_shape)
        gate_1 = self.gate_1.expand(batch_shape)
        base_dist = self.base_dist.expand(batch_shape)
        new.__init__(base_dist, gate_0=gate_0, gate_1=gate_1, validate_args=False)
        new._validate_args = self._validate_args
        return new


class ZeroOneAdjustedBeta(ZeroOneInflatedDistribution):
    """
    A Zero and One Adjusted Beta distribution.

    This distribution assigns probability mass at 0 and 1 (via gate_0 and gate_1, respectively)
    while modeling values in (0, 1) with a Beta distribution.

    Parameters
    ----------
    concentration1 : torch.Tensor
        1st concentration (alpha) parameter of the Beta component.
    concentration0 : torch.Tensor
        2nd concentration (beta) parameter of the Beta component.
    gate_0 : torch.Tensor or None
        Probability mass for 0.
    gate_0_logits : torch.Tensor or None
        Logits for probability mass for 0.
    gate_1 : torch.Tensor or None
        Probability mass for 1.
    gate_1_logits : torch.Tensor or None
        Logits for probability mass for 1.
    validate_args : bool, optional
        Whether to validate input arguments.
    """
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
        "gate_0": constraints.unit_interval,
        "gate_1": constraints.unit_interval,
    }
    support = constraints.unit_interval

    def __init__(self, concentration1, concentration0, 
                 gate_0=None, gate_0_logits=None, 
                 gate_1=None, gate_1_logits=None, 
                 validate_args=None):
        base_dist = Beta(concentration1=concentration1, concentration0=concentration0, validate_args=False)
        base_dist._validate_args = validate_args
        super().__init__(base_dist,
                         gate_0=gate_0, gate_0_logits=gate_0_logits,
                         gate_1=gate_1, gate_1_logits=gate_1_logits,
                         validate_args=validate_args)

    @property
    def concentration1(self):
        return self.base_dist.concentration1

    @property
    def concentration0(self):
        return self.base_dist.concentration0
