from torchsim_AL import sampler_spec
import pytest
from torch_sim.units import MetalUnits

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

def test_unsorted_temperatures_are_rejected():
    with pytest.raises(ValueError, match="increasing"):
        sampler_spec.validate_spec(**_base_kwargs(temperatures=[300.0, 500.0, 400.0]))

def test_one_temp_and_apply_swaps():
    with pytest.raises(ValueError, matches="At least 2"):
        sampler_spec.validate_spec(**_base_kwargs(temperatures=[300.0, 500.0, 400.0], swap_temperature=True))

def test_nvt_with_unset_pressure_is_accepted():
    spec = sampler_spec.validate_spec(**_base_kwargs())
    assert spec["external_pressure"] is None

def test_nvt_with_set_pressure_rejected():
    with pytest.raises(ValueError, matches="must be None or 0"):
        sampler_spec.validate_spec(**_base_kwargs(external_pressure=1))

def test_npt_without_set_pressure():
    spec = sampler_spec.validate_spec(**_base_kwargs(integrator_name="npt_langevin_isotropic", step_kwargs={}, init_kwargs={"b_tau": 1.0, "alpha": 1.0}))
    assert spec["external_pressure"] == 0
    assert spec["init_kwargs"] == {"b_tau": 1.0, "alpha": 1.0}

def test_integrator_defaults():
    spec = sampler_spec.validate_spec(**_base_kwargs(step_kwargs={}))
    assert spec["temperatures"] == [300.0, 400.0, 500.0]
    assert spec["integrator_name"] == "nvt_langevin"
    assert spec["dt"] == 0.001
    assert spec["step"] == {"gamma": 1/(100*spec["dt"]*MetalUnits.time)}

def test_unknown_step_kwarg_is_rejected_with_specific_message():
    with pytest.raises(ValueError, match="not accepted"):
        sampler_spec.validate_spec(**_base_kwargs(step_kwargs={"not_a_real_kwarg": 1}))

def test_reserved_kwarg_in_step_kwargs_is_rejected_with_specific_message():
    with pytest.raises(ValueError, match="supplied automatically"):
        sampler_spec.validate_spec(**_base_kwargs(integrator_name="npt_langevin_isotropic", step_kwargs={"external_pressure": 1.0}))

def test_multiple_errors_are_all_reported_together():
    with pytest.raises(ValueError) as exc_info:
        sampler_spec.validate_spec(**_base_kwargs(
            dt=-1,
            init_kwargs = {"gamma": 1.0},
            model_paths=[],
        ))
    message = str(exc_info.value)
    assert "dt" in message
    assert "kwarg" in message
    assert "model_paths" in message
