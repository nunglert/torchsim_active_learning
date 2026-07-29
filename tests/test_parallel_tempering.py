import pytest
import torch
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
def replica_atoms(small_structure):
    small_structure = Atoms(symbols=["Ar"] * 1, positions=(1, 2, 3), cell=[10, 10, 10], pbc=True)
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
        n_steps_npt=0,
        n_steps_pt=20,
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