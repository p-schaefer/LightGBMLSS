from .zero_one_inflated import ZeroOneAdjustedDirichlet as ZeroOneAdjustedDirichlet_Torch
from .distribution_utils import DistributionClass
from ..utils import *

class ZOADirichlet(DistributionClass):
    r"""
    Zero and One Adjusted Dirichlet distribution class.

    This distribution models outcomes on a K-dimensional simplex with the added
    flexibility of placing extra probability mass at the vertices. In other words,
    a portion of the probability mass is assigned directly to one-hot vectors (vertices)
    while the remainder is modeled with a Dirichlet distribution.

    Distributional Parameters
    -------------------------
    concentration : torch.Tensor
        A vector of concentration parameters (shape: (..., K)) for the Dirichlet component.
    gate : torch.Tensor
        A gating vector (shape: (..., K)) whose components lie in [0,1]. Their sum must be ≤ 1.
        They represent the probability mass placed at each vertex.

    The probability of drawing from the continuous component is
          1 - sum(gate).

    Parameters
    -------------------------
    stabilization : str, default "None"
        Stabilization method for the gradient and Hessian. Options: "None", "MAD", "L2".
    response_fn : str, default "exp"
        Response function for transforming distributional parameters to the correct support.
        Options: "exp" or "softplus".
    loss_fn : str, default "nll"
        Loss function. Options: "nll" (negative log-likelihood).
    """
    def __init__(self,
                 stabilization: str = "None",
                 response_fn: str = "exp",
                 loss_fn: str = "nll"):
        # Input Checks.
        if stabilization not in ["None", "MAD", "L2"]:
            raise ValueError("Invalid stabilization method. Please choose from 'None', 'MAD' or 'L2'.")
        if loss_fn not in ["nll"]:
            raise ValueError("Invalid loss function. Please select 'nll'.")

        # Specify response functions.
        response_functions = {"exp": exp_fn, "softplus": softplus_fn}
        if response_fn in response_functions:
            response_fn_callable = response_functions[response_fn]
        else:
            raise ValueError("Invalid response function. Please choose from 'exp' or 'softplus'.")

        # Set the Torch distribution to our Zero and One Adjusted Dirichlet.
        distribution = ZeroOneAdjustedDirichlet_Torch
        # Define which parameters are expected and their corresponding transformation.
        param_dict = {
            "concentration": response_fn_callable,
            "gate": sigmoid_fn,
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        super().__init__(distribution=distribution,
                         univariate=False,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )
                         
