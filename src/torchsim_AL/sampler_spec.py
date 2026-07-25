from typing import Any, Optional
from torchsim_AL.md_kwargs import MDSpec

def validate_spec(
    *,
    founder: str,
    integrator_name: str,
    dt: float=0.001,
    init_kwargs: dict[str, Any],
    step_kwargs: dict[str, Any],
    temperatures: list[float],
    model_paths: list[str],
    external_pressure: Optional[float] = None,
    n_steps_npt: int = 0,
    n_steps_pt: int = 100,
    seed: int = 123,
    device: str = "cuda",
    dtype: str = "float32",
    md_steps_per_exchange: int = 100,
) -> dict[str, Any]:
    """Validate all sampler inputs and return the dict written to input_torchsim.json."""

    # integrator kwargs validated against torchsim's real function signatures
    md_spec = MDSpec(integrator_name=integrator_name, init_kwargs=init_kwargs, step_kwargs=step_kwargs)
    if integrator_name.startswith("nvt"):
        if external_pressure!=0:
            raise ValueError(f"no pressure can be set for NVT simulations. Got: {external_pressure}")
    if not temperatures or any(t <= 0 for t in temperatures):
        raise ValueError(f"temperatures must be non-empty and positive. Got: {temperatures}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n_steps_pt <= 0:
        raise ValueError(f"n_steps_pt must be bigger than 0, got {n_steps_pt}")
    if n_steps_npt < 0:
        raise ValueError(f"n_steps_pt must be positive, got {n_steps_npt}")
    if not model_paths:
        raise ValueError("At least one model path must be provided.")
    if "foundation" in model_paths and len(model_paths) > 1:
        raise ValueError(f"'foundation' cannot be combined with other model paths: {model_paths}")
    if device not in ("cuda", "cpu"):
        raise ValueError(f"Unknown device '{device}'.")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"Unknown dtype '{dtype}'.")
    if n_steps_pt <= 0:
        raise ValueError("n_steps_pt must be greater than 0.")

    return {
        **md_spec.to_dict(),
        "founder": founder,
        "temperatures": temperatures,
        "model_paths": model_paths,
        "external_pressure": external_pressure,
        "n_steps_npt": n_steps_npt,
        "n_steps_pt": n_steps_pt,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "md_steps_per_exchange": md_steps_per_exchange,
        "model_paths": model_paths,
        "init_kwargs": init_kwargs,
        "step_kwargs": step_kwargs
    }