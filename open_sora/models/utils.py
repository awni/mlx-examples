import mlx.nn as nn

def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """
    ACTIVATION_FUNCTIONS = {
        "swish": nn.SiLU(),
        "silu": nn.SiLU(),
        "mish": nn.Mish(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
    }
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
