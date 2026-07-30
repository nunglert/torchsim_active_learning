"""Validation for everything that goes into a torch_sim PT sampling job.

``validate_spec`` is the single choke point all sampler inputs pass through
before anything is written to disk or copied to the cluster (see
``TorchSimSampler.update``). It never mutates its inputs; on success it
returns a plain, JSON-serializable dict matching exactly what
``torch_sim_sampling.py`` expects to find in the input file.

Rather than raising on the first problem found, we collect
*all* problems and raise once with the full list, before anything gets put on the cluster.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from torchsim_AL.md_kwargs import MDSpec

logger = logging.getLogger(__name__)

_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")
_VALID_DTYPES = ("float32", "float64")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_int(errors: list[str], name: str, value: Any, *, strict_positive: bool = False, positive: bool = False) -> bool:
    """Validate an int field. Returns True if valid (safe to use in further checks).
    strict_positive=True requires value > 0; positive=True requires value >= 0"""
    if not _is_int(value):
        errors.append(f"'{name}' must be an int, got {type(value).__name__} ({value!r}).")
        return False
    if strict_positive and value <= 0:
        errors.append(f"'{name}' must be > 0, got {value}.")
        return False
    if positive and value < 0:
        errors.append(f"'{name}' must be >= 0, got {value}.")
        return False
    return True


def validate_spec(
    *,
    founder: str,
    integrator_name: str,
    dt: float = 0.001,
    init_kwargs: dict[str, Any],
    step_kwargs: dict[str, Any],
    temperatures: list[float],
    model_paths: list[str],
    external_pressure: Optional[float] = None,
    n_steps_equil: int = 0,
    n_steps_prod: int = 100,
    seed: int = 42,
    device: str = "cuda",
    dtype: str = "float64",
    md_steps_per_exchange: int = 100,
    write_frequency: int = 10,
    attempt_swap: bool = True,
    rescale_momenta_on_swap: bool = True,
    rescale_barostat_on_swap: bool = True,
) -> dict[str, Any]:
    """Validate all sampler inputs and return the dict written to input_torchsim.json.

    Notes on units (see README for the full table): ``temperatures`` in K,
    ``dt`` in ps, ``external_pressure`` in bar. All other init_kwargs/
    step_kwargs (tau, gamma, alpha, cell_alpha, b_tau, t_tau, tau_p,
    isothermal_compressibility, ...) are passed through to torch_sim
    UNCONVERTED, i.e. in torch_sim's native "metal" unit system, not in
    physical ps/ps^-1/bar^-1. See the README for the exact conversion.

    Raises:
        ValueError: if one or more inputs are invalid. The message lists
            every problem found, not just the first.
    """
    errors: list[str] = []

    # -- founder --------------------------------------------------------
    if not isinstance(founder, str) or not founder:
        errors.append(f"'founder' must be a non-empty string (filenpath to .extxyz/.xyz file), got {founder!r}.")

    # -- integrator_name / init_kwargs / step_kwargs ---------------------
    kwargs_types_ok = True
    if not isinstance(init_kwargs, dict):
        errors.append(f"'init_kwargs' must be a dict, got {type(init_kwargs).__name__}.")
        kwargs_types_ok = False
    if not isinstance(step_kwargs, dict):
        errors.append(f"'step_kwargs' must be a dict, got {type(step_kwargs).__name__}.")
        kwargs_types_ok = False

    if kwargs_types_ok:
        try:
            md_spec = MDSpec(
                integrator_name=integrator_name, init_kwargs=init_kwargs, step_kwargs=step_kwargs
            )
        except (ValueError, TypeError) as exc:
            md_spec = None
            errors.append(str(exc))
    else:
        md_spec = None

    # -- temperatures -----------------------------------------------------
    if not temperatures:
        errors.append("'temperatures' must be a non-empty list.")
    elif not all(_is_number(t) for t in temperatures):
        errors.append(f"'temperatures' must all be numeric (K), got: {temperatures}")
    elif any(t <= 0 for t in temperatures):
        errors.append(f"'temperatures' must all be positive (K), got: {temperatures}")
    elif len(temperatures) < 2 and attempt_swap == True:
        errors.append(
            f"At least 2 temperatures are required to do replica exchange, got {len(temperatures)}."
        )
    elif any(t2 <= t1 for t1, t2 in zip(temperatures, temperatures[1:])):
        errors.append(
            "'temperatures' must be strictly increasing: replica exchange only "
            "attempts swaps between neighbor-indexed replicas (i, i+1), so an "
            f"unsorted or duplicate ladder silently breaks that pairing. Got: {temperatures}"
        )

    # -- dt -----------------------------------------------------------------
    if not _is_number(dt):
        errors.append(f"'dt' must be numeric (ps), got {type(dt).__name__} ({dt!r}).")
    elif dt <= 0:
        errors.append(f"'dt' must be positive (ps), got {dt}.")
    elif dt > 0.01:
        logger.warning(
            "dt=%.4g ps is unusually large for atomistic MD (typical range is "
            "0.0002-0.002 ps / 0.2-2 fs). Double check this is intentional.",
            dt,
        )

    # -- external_pressure vs. ensemble -------------------------------------
    if external_pressure is not None and not _is_number(external_pressure):
        errors.append(
            f"'external_pressure' must be numeric (bar) or None, got "
            f"{type(external_pressure).__name__} ({external_pressure!r})."
        )
    elif isinstance(integrator_name, str) and integrator_name.startswith("nvt"):
        if external_pressure not in (None, 0, 0.0):
            errors.append(
                f"'external_pressure' must be None or 0 for NVT integrators "
                f"(no barostat) — got {external_pressure!r} with integrator_name="
                f"'{integrator_name}'."
            )
    elif isinstance(integrator_name, str) and integrator_name.startswith("npt"):
        if external_pressure is None:
            logger.warning(
                "external_pressure not set for NPT integrator '%s' — defaulting "
                "to 0 bar. Pass external_pressure explicitly if that's not intended.",
                integrator_name,
            )

    # -- step counts / cadence ------------------------------------------------
    n_steps_prod_ok = _check_int(errors, "n_steps_prod", n_steps_prod, strict_positive=True)
    n_steps_equil_ok = _check_int(errors, "n_steps_equil", n_steps_equil, positive=True)
    md_steps_per_exchange_ok = _check_int(
        errors, "md_steps_per_exchange", md_steps_per_exchange, strict_positive=True
    )
    write_frequency_ok = _check_int(errors, "write_frequency", write_frequency, strict_positive=True)

    if n_steps_prod_ok and write_frequency_ok and write_frequency > n_steps_prod:
        errors.append(
            f"'write_frequency' ({write_frequency}) is larger than 'n_steps_prod' "
            f"({n_steps_prod}) — no trajectory frames would ever be written."
        )
    elif n_steps_prod_ok and write_frequency_ok and n_steps_prod % write_frequency != 0:
        logger.warning(
            "n_steps_prod (%d) is not a multiple of write_frequency (%d); the final "
            "partial block will not be written to the trajectory.",
            n_steps_prod,
            write_frequency,
        )
    if (
        n_steps_prod_ok
        and md_steps_per_exchange_ok
        and md_steps_per_exchange > n_steps_prod
    ):
        logger.warning(
            "md_steps_per_exchange (%d) > n_steps_prod (%d): no replica-exchange "
            "swaps will ever be attempted, this reduces to independent MD per "
            "replica.",
            md_steps_per_exchange,
            n_steps_prod,
        )

    # -- seed -----------------------------------------------------------------
    if not _is_int(seed):
        errors.append(f"'seed' must be an int, got {type(seed).__name__} ({seed!r}).")

    # -- model_paths ------------------------------------------------------
    if not model_paths:
        errors.append("'model_paths' must be a non-empty list.")
    elif not all(isinstance(p, str) for p in model_paths):
        errors.append(f"'model_paths' must all be strings, got: {model_paths}")
    elif "foundation" in model_paths and len(model_paths) > 1:
        errors.append(
            f"'foundation' cannot be combined with other model paths: {model_paths}"
        )

    # -- device / dtype -----------------------------------------------------
    if not isinstance(device, str) or not _DEVICE_RE.match(device):
        errors.append(f"'device' must be 'cpu', 'cuda', or 'cuda:N', got {device!r}.")
    if dtype not in _VALID_DTYPES:
        errors.append(f"'dtype' must be one of {_VALID_DTYPES}, got {dtype!r}.")

    # -- swap behavior flags --------------------------------------------------
    for name, value in (
        ("attempt_swap", attempt_swap),
        ("rescale_momenta_on_swap", rescale_momenta_on_swap),
        ("rescale_barostat_on_swap", rescale_barostat_on_swap),
    ):
        if not isinstance(value, bool):
            errors.append(f"'{name}' must be a bool, got {type(value).__name__} ({value!r}).")

    if errors:
        message = "Invalid sampler configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(message)
        raise ValueError(message)

    return {
        **md_spec.to_dict(),
        "founder": founder,
        "dt": dt,
        "temperatures": temperatures,
        "model_paths": model_paths,
        "external_pressure": external_pressure,
        "n_steps_equil": n_steps_equil,
        "n_steps_prod": n_steps_prod,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "md_steps_per_exchange": md_steps_per_exchange,
        "write_frequency": write_frequency,
        "attempt_swap": attempt_swap,
        "rescale_momenta_on_swap": rescale_momenta_on_swap,
        "rescale_barostat_on_swap": rescale_barostat_on_swap,
    }