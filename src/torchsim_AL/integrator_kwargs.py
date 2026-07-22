import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch_sim as ts
from torch_sim.integrators import INTEGRATOR_REGISTRY

logger = logging.getLogger(__name__)


def _validate_kwargs(sig: inspect.Signature, kwargs: dict, label: str, ignore: set[str]) -> tuple[list[str], list[str]]:
    params = {name: p for name, p in sig.parameters.items() if name not in ignore}

    unknown = set(kwargs) - set(params)
    if unknown:
        raise TypeError(f"[{label}] Unknown parameter(s): {sorted(unknown)}. Valid: {sorted(params)}")

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
        init_sig = inspect.signature(self.init_func)
        step_sig = inspect.signature(self.step_func)

        missing_init, defaulted_init = _validate_kwargs(
            init_sig, self.init_kwargs, label=f"{self.integrator_name}.init", ignore={"state", "model", "dt", "kT"}
        )
        missing_step, defaulted_step = _validate_kwargs(
            step_sig, self.step_kwargs, label=f"{self.integrator_name}.step", ignore={"state", "model", "dt", "kT"}
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