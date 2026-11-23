from dnqsm.model import ModelConfig


def laplacian_model_config(*args, **kwargs):
    """Explicit historical architecture for Laplacian compatibility regressions."""
    kwargs.setdefault("filter_input_version", 2)
    kwargs.setdefault("cutoff_mode", "hard")
    kwargs.setdefault("laplacian_input", True)
    kwargs.setdefault("conditional_laplacian_threshold", 32.0)
    kwargs.setdefault("downsampling", "max_pool")
    kwargs.setdefault("bias", True)
    kwargs.setdefault("laplacian_operator", "stencil")
    kwargs.setdefault("laplacian_extension_iters", 0)
    kwargs.setdefault("laplacian_extension_mode", "harmonic")
    return ModelConfig(*args, **kwargs)


def dual_field_model_config(*args, **kwargs):
    """Explicit opt-in to the experimental raw-input dual-field architecture."""
    kwargs.setdefault("filter_input_version", 3)
    kwargs.setdefault("cutoff_mode", "hard")
    kwargs.setdefault("laplacian_input", False)
    kwargs.setdefault("conditional_laplacian_threshold", 0.0)
    kwargs.setdefault("downsampling", "max_pool")
    kwargs.setdefault("bias", True)
    kwargs.setdefault("laplacian_operator", "stencil")
    kwargs.setdefault("laplacian_extension_iters", 0)
    kwargs.setdefault("laplacian_extension_mode", "harmonic")
    return ModelConfig(*args, **kwargs)
