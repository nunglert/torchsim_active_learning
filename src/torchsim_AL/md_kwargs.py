import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch_sim as ts
from torch_sim.integrators import INTEGRATOR_REGISTRY

logger = logging.getLogger(__name__)

RESERVED_KWARGS: set[str] = {"state", "model", "dt", "kT", "external_pressure"}


def _validate_kwargs(
    sig: inspect.Signature, kwargs: dict, label: str, ignore: set[str]
) -> tuple[list[str], list[str]]:
    """Check kwargs against a function signature, ignoring reserved names.

    Args:
        sig: Signature of the init/step function being validated against.
        kwargs: The init_kwargs or step_kwargs dict supplied by the caller.
        label: Human-readable tag for error messages, e.g. "nvt_langevin.step".
        ignore: Parameter names to exclude from validation — these are
            supplied programmatically elsewhere (see RESERVED_KWARGS) and
            shouldn't be flagged as "missing" just because the caller didn't
            supply them here.

    Returns:
        (missing, defaulted): ``missing`` are required parameters (no
        default) the caller didn't supply — this is fatal. ``defaulted``
        are optional parameters the caller didn't supply, so torch_sim will
        fall back to its own default — this is only logged as a warning.

    Raises:
        TypeError: if kwargs contains a name the function doesn't accept,
            distinguishing "reserved, supplied automatically" from "not a
            real parameter of this function" in the message.
    """

    params = {name: p for name, p in sig.parameters.items() if name not in ignore}

    unknown = set(kwargs) - set(params)
    if unknown:
        reserved_hit = unknown & RESERVED_KWARGS
        real_unknown = unknown - RESERVED_KWARGS
        msgs = []
        if reserved_hit:
            msgs.append(
                f"{sorted(reserved_hit)} are supplied automatically by the "
                "sampler and must not be set in init_kwargs/step_kwargs"
            )
        if real_unknown:
            msgs.append(
                f"{sorted(real_unknown)} are not accepted by this function. "
                f"Valid: {sorted(params)}"
            )
        raise TypeError(f"[{label}] " + "; ".join(msgs))

    missing = [
        name for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in kwargs
    ]

    defaulted = [
        name for name, p in params.items()
        if p.default is not inspect.Parameter.empty
        and name not in kwargs
    ]

    return missing, defaulted


@dataclass
class MDSpec:
    """One integrator's init/step kwargs, validated against torch_sim's real
        function signatures on construction.

        Names that are supplied programmatically by the sampler / parallel_tempering
        loop and must never be set by the user via init_kwargs or step_kwargs.

        - state, model, dt, kT: threaded through explicitly on every init/step call.
        - external_pressure: injected into step_kwargs by parallel_tempering() from
          the sampler's top-level `external_pressure` field (see sampler_spec.py)

        Resolving ``integrator_name`` and validating ``init_kwargs``/
        ``step_kwargs`` both happen in ``__post_init__``, so an invalid
        integrator name or a bad kwarg raises immediately when the object is
        built — not later when the job actually runs on the cluster.
        """

    integrator_name: str
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    step_kwargs: dict[str, Any] = field(default_factory=dict)

    init_func: Callable = field(init=False, repr=False, compare=False)
    step_func: Callable = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        try:
            integrator = getattr(ts.Integrator, self.integrator_name)
        except AttributeError:
            valid = [m.name for m in ts.Integrator]
            raise ValueError(f"Unknown integrator '{self.integrator_name}'. Available: {valid}") from None

        self.init_func, self.step_func = INTEGRATOR_REGISTRY[integrator]
        self.validate_spec()

    def validate_spec(self) -> None:
        """Validate init_kwargs/step_kwargs against the resolved integrator's
        real signatures.

        Not to be confused with ``sampler_spec.validate_spec`` (module-level,
        validates the whole sampler config) — this only checks this one
        integrator's kwargs, and is called automatically from __post_init__.

        Raises:
            TypeError: if either dict has a missing required parameter or an
                unrecognized one.
        """
        init_sig = inspect.signature(self.init_func)
        step_sig = inspect.signature(self.step_func)

        missing_init, defaulted_init = _validate_kwargs(
            init_sig, self.init_kwargs, label=f"{self.integrator_name}.init", ignore=RESERVED_KWARGS
        )
        missing_step, defaulted_step = _validate_kwargs(
            step_sig, self.step_kwargs, label=f"{self.integrator_name}.step", ignore=RESERVED_KWARGS
        )

        if missing_init or missing_step:
            raise TypeError(
                f"[{self.integrator_name}] missing required parameter(s) — "
                f"init: {missing_init}, step: {missing_step}"
            )

        if defaulted_init:
            logger.warning(f"[{self.integrator_name}] init falling back to defaults for: {defaulted_init}")
        if defaulted_step:
            logger.warning(f"[{self.integrator_name}] step falling back to defaults for: {defaulted_step}")

    def to_dict(self) -> dict[str, Any]:
        """Only plain, JSON-safe input data — derived functions are excluded."""
        return {
            "integrator_name": self.integrator_name,
            "init_kwargs": self.init_kwargs,
            "step_kwargs": self.step_kwargs,
        }