import torch
from torch.distributions import constraints
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus

from torch.distributions import NegativeBinomial, Poisson, Gamma, LogNormal, Beta, Dirichlet
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
      
class ZeroOneAdjustedDirichlet(TorchDistribution):
    r"""
    A Zero and One Adjusted Dirichlet distribution.

    This distribution models outcomes on a K-dimensional simplex but assigns
    extra mass at the vertices. That is, given K (Dirichlet) concentration parameters
    and a vector of gate probabilities (one for each vertex) with
        sum(gate) ≤ 1,
    the overall model is a mixture:
    
      • With probability gate[i] the outcome is the i-th vertex (a one-hot vector),
      • Otherwise (with probability 1 – sum(gate)) the outcome is drawn from a Dirichlet
        with the given concentration parameters.
    
    Parameters
    ----------
    concentration : torch.Tensor
        Concentration parameters (shape … x K) for the Dirichlet component.
    gate : torch.Tensor, optional
        A tensor (shape … x K) of probabilities for each vertex. Must satisfy 0 ≤ gate_i ≤ 1
        and sum(gate) ≤ 1.
    gate_logits : torch.Tensor, optional
        Alternative to gate. If provided, these logits are converted to probabilities
        using torch.distributions’ logits_to_probs.
    validate_args : bool, optional
        Whether to validate input arguments.
    """
    arg_constraints = {
        "concentration": constraints.positive,
        # Note: there is no built-in constraint for a vector whose elements lie in [0,1]
        # and whose sum is at most 1. Users must ensure this.
    }
    support = constraints.simplex

    def __init__(self, concentration, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError("Either `gate` or `gate_logits` must be specified, but not both.")
        
        # Broadcast concentration and gate appropriately.
        concentration = torch.as_tensor(concentration)
        if gate is not None:
            gate = torch.as_tensor(gate)
            self._gate, concentration = broadcast_all(gate, concentration)
        else:
            # Use provided logits to get gate probabilities.
            gate_from_logits = logits_to_probs(torch.as_tensor(gate_logits))
            self._gate, concentration = broadcast_all(gate_from_logits, concentration)
        
        # Ensure that the sum along the last dimension does not exceed 1.
        if (self._gate.sum(dim=-1) > 1).any():
            raise ValueError("The sum of gate values must be less than or equal to 1.")
        
        self.base_dist = Dirichlet(concentration, validate_args=validate_args)
        batch_shape = self.base_dist.batch_shape
        event_shape = self.base_dist.event_shape  # Should be (K,)
        super().__init__(batch_shape, event_shape, validate_args)

    @lazy_property
    def gate(self):
        return self._gate

    @lazy_property
    def continuous_weight(self):
        # The weight of the continuous (Dirichlet) component.
        return 1 - self.gate.sum(dim=-1)

    def log_prob(self, value):
        """
        Computes the log density of an observation.

        For an observation (vector) x:
          • If x is a vertex, i.e. one coordinate is nearly 1 and the rest nearly 0,
            returns log(gate_i) for that vertex.
          • Otherwise, returns log(continuous_weight) + log_prob(x) under Dirichlet.
        """
        if self._validate_args:
            self._validate_sample(value)
            
        eps = abs(torch.finfo(value.dtype).eps)
        # Clamp the input to avoid numerical issues for the continuous part.
        value_cont = value.clamp(eps, 1 - eps)
        
        # Identify vertices. A vertex has one coordinate ~1 and the rest ~0.
        is_one = value >= (1 - eps)
        is_zero = value <= eps
        # For each sample, if exactly one coordinate is nearly one and the rest nearly zero.
        vertex_indicator = (is_one.sum(dim=-1) == 1) & (is_zero.sum(dim=-1) == (value.size(-1) - 1))
        
        # Compute continuous (Dirichlet) log density.
        lp_cont = torch.log(self.continuous_weight + eps) + self.base_dist.log_prob(value_cont)
        
        # For vertex samples, get the corresponding gate log-probability.
        vertex_idx = value.argmax(dim=-1)  # (batch,)-shaped indices.
        lp_vertex = torch.log(self.gate.gather(dim=-1, index=vertex_idx.unsqueeze(-1)) + eps).squeeze(-1)
        
        # Combine: use the vertex log-probability where appropriate.
        logp = torch.where(vertex_indicator, lp_vertex, lp_cont)
        return logp

    def sample(self, sample_shape=torch.Size()):
        """
        Draw samples from the mixture distribution.

        For each draw, a categorical decision is made:
          • With probability gate[i] (for some i) the outcome is the vertex with 1 at
            coordinate i.
          • With probability continuous_weight the outcome is drawn from the Dirichlet.
        """
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            # Form the mixture probabilities: a vector of length K+1.
            # The first K entries are the gate-values; the last is the continuous component weight.
            p_gate = self.gate.expand(shape)
            p_cont = self.continuous_weight.expand(shape).unsqueeze(-1)
            mixture_probs = torch.cat([p_gate, p_cont], dim=-1)
            # Sample from a categorical distribution over {0,...,K} (K outcomes for vertices,
            # outcome K for the continuous Dirichlet component).
            cat = torch.distributions.Categorical(mixture_probs)
            mixture_idx = cat.sample()  # Shape: shape
            # Sample continuous outcomes.
            cont_sample = self.base_dist.expand(shape).sample()
            # For discrete outcomes, produce the corresponding one-hot vector.
            k = self.gate.size(-1)
            disc_sample = torch.nn.functional.one_hot(mixture_idx.clamp(max=k-1), num_classes=k).to(cont_sample.dtype)
            # Combine: if mixture_idx == k, choose continuous sample; otherwise, use discrete one-hot.
            mask = (mixture_idx == k).unsqueeze(-1)
            sample = torch.where(mask, cont_sample, disc_sample)
        return sample

    @lazy_property
    def mean(self):
        """
        Computes the overall mixture mean.

        For discrete outcomes the mean is the one-hot vector (i.e. the vector of gate values when averaged).
        Overall:
          E[X] = (∑ₖ gateₖ * one_hotₖ) + continuous_weight * (Dirichlet mean)
               = gate + continuous_weight.unsqueeze(-1) * base_dist.mean.
        """
        return self.gate + self.continuous_weight.unsqueeze(-1) * self.base_dist.mean

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        # Expand concentration and gate.
        concentration = self.base_dist.concentration.expand(batch_shape + self.base_dist.event_shape)
        gate = self.gate.expand(batch_shape + self.gate.shape[-1:])
        new.__init__(concentration, gate=gate, validate_args=False)
        new._validate_args = self._validate_args
        return 
