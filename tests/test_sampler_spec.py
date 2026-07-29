from torchsim_AL import sampler_spec
import pytest

def _base_kwargs(**overrides):
    """A minimal, otherwise-valid config; tests override one field at a time."""
    kwargs = dict(
        founder="founder.extxyz",
        integrator_name="nvt_langevin",
        dt=0.001,
        init_kwargs={},
        step_kwargs={"gamma": 1.0},
        temperatures=[300.0, 400.0, 500.0],
        model_paths=["foundation"],
    )
    kwargs.update(overrides)
    return kwargs

def test_valid_config_is_accepted():
    spec = sampler_spec.validate_spec(**_base_kwargs())
    assert spec["temperatures"] == [300.0, 400.0, 500.0]
    assert spec["integrator_name"] == "nvt_langevin"
    assert spec["dt"] == 0.001
    assert spec["step"] == {"gamma": 1.0}

