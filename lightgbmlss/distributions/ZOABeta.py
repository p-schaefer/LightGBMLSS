from .zero_one_inflated import ZeroOneAdjustedBeta as ZeroOneAdjustedBeta_Torch
from .distribution_utils import DistributionClass
from ..utils import *


class ZOBeta(DistributionClass):
    """
    Zero and One Adjusted Beta distribution class.

    Distributional Parameters
    -------------------------
    concentration1 : torch.Tensor
        1st concentration parameter (alpha) of the Beta component.
    concentration0 : torch.Tensor
        2nd concentration parameter (beta) of the Beta component.
    gate_0 : torch.Tensor
        Probability mass at 0.
    gate_1 : torch.Tensor
        Probability mass at 1.

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
        # Input checks.
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

        # Set the Torch distribution to our Zero and One Adjusted Beta.
        distribution = ZeroOneAdjustedBeta_Torch
        param_dict = {
            "concentration1": response_fn_callable,
            "concentration0": response_fn_callable,
            "gate_0": sigmoid_fn,
            "gate_1": sigmoid_fn
        }
        torch.distributions.Distribution.set_default_validate_args(False)

        super().__init__(distribution=distribution,
                         univariate=True,
                         discrete=False,
                         n_dist_param=len(param_dict),
                         stabilization=stabilization,
                         param_dict=param_dict,
                         distribution_arg_names=list(param_dict.keys()),
                         loss_fn=loss_fn
                         )
