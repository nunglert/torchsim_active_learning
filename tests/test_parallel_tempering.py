import pytest
import torch
from torch_sim.units import MetalUnits
from torch_sim import TorchSimTrajectory
from torchsim_AL.scripts.parallel_tempering import parallel_tempering
from torchsim_AL.scripts.uncertainty_props import _build_prop_calculators
from ase import Atoms
import numpy as np
import copy

@pytest.fixture
def lj_model():
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(
        sigma=3.4, epsilon=0.01, device=torch.device("cpu"), dtype=torch.float64,
        compute_forces=True, compute_stress=True, cutoff=6.0,
    )

@pytest.fixture
def replica_atoms():
    small_structure = Atoms(symbols=["Ar"] * 1, positions=[(1, 2, 3)], cell=[10, 10, 10], pbc=True)
    return [copy.deepcopy(small_structure) for _ in range(3)]

def _run(replica_atoms, lj_model, tmp_path, **overrides):
    kwargs = dict(
        replica_atoms=replica_atoms,
        model=lj_model,
        integrator_name="nvt_langevin",
        temperatures=[100.0, 150.0, 220.0],
        dt=0.001,
        init_kwargs={},
        step_kwargs={"gamma": 0.5},
        dtype=torch.float64,
        device=torch.device("cpu"),
        n_steps_npt=10,
        n_steps_pt=30,
        md_steps_per_exchange=5,
        seed=42,
    )
    kwargs.update(overrides)

    filenames = [tmp_path / f"trajectory_{i}.h5md" for i in range(len(replica_atoms))]
    reporter_args = dict(
        filenames=filenames,
        state_frequency=4,
        prop_calculators=_build_prop_calculators(lj_model, frequency=4),
    )
    return parallel_tempering(**kwargs, reporter_args=reporter_args), filenames

def test_default_settings_run_without_crashing(replica_atoms, lj_model, tmp_path):
    (state, temperatures, P, swap_log), _ = _run(replica_atoms, lj_model, tmp_path)
    assert swap_log["info"]["attempted"] > 0

def test_swap_temperature_false_runs_without_crashing(replica_atoms, lj_model, tmp_path):
    state, temperatures, P, swap_log, _ = _run(replica_atoms, lj_model, tmp_path, swap_temperature=False)
    assert swap_log["info"]["attempted"] == 0
    assert swap_log["info"]["accepted"] == 0

def test_npt_runs_without_crashing(replica_atoms, lj_model, tmp_path):
    (state, temperatures, P, swap_log), _ = _run(
        replica_atoms, lj_model, tmp_path,
        integrator_name="npt_langevin_isotropic", external_pressure=1.0, step_kwargs={},
    )
    assert swap_log["info"]["external_pressure"] == pytest.approx(1.0 * MetalUnits.pressure)

def test_same_seed_is_fully_reproducible(replica_atoms, lj_model, tmp_path):
    (state_a, _, _, log_a), _ = _run(replica_atoms, lj_model, tmp_path, seed=7)
    (state_b, _, _, log_b), _ = _run(replica_atoms, lj_model, tmp_path, seed=7)
    assert log_a["data"] == log_b["data"]
    assert state_a == state_b

def test_total_steps_is_equilibration_plus_production(replica_atoms, lj_model, tmp_path):
    (state_a, _, _, log_a), _ = _run(replica_atoms, lj_model, tmp_path)
    filenames = [tmp_path / f"trajectory_{i}.h5md" for i in range(len(replica_atoms))]
    traj = TorchSimTrajectory(str(filenames[0]), mode="r")
    last_recorded_step = int(traj.get_steps("positions")[-1])
    traj.close()
    assert last_recorded_step == 40