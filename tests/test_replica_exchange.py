import pytest
from torchsim_AL.scripts.replica_exchange import attempt_replica_exchange
import torch
import copy
from ase import Atoms
import torch_sim as ts
from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits
from torch_sim.models.lennard_jones import LennardJonesModel

def _replica_exchange_kwargs(**overrides):
    kwargs=dict(
        i = 0,
        j = 1,
        rng = torch.Generator(device="cpu").manual_seed(42),
        kT = torch.Tensor([100, 200]),
        external_pressure = 1 * MetalUnits.pressure,
        rescale_momenta = True,
        rescale_barostat = True, 
    )
    kwargs.update(overrides)
    return kwargs

model = LennardJonesModel(
    sigma=3.4, epsilon=0.01, device=torch.device("cpu"), dtype=torch.float64,
    compute_forces=True, compute_stress=True, cutoff=6.0,
)

def _create_md_state(seed, temps):
    small_structure = Atoms(symbols=["Ar"]*2, positions=[(1, 2, 3), (3, 2, 1)], cell=[10, 10, 10], pbc=True)
    replica_atoms = [copy.deepcopy(small_structure) for _ in range(2)]
    initial_state = ts.initialize_state(replica_atoms, torch.device("cpu"), torch.float64)
    initial_state.rng = seed
    state = nvt_langevin_init(state=initial_state, model=model, kT=temps, dt=0.001)
    state = nvt_langevin_step(state=state, model=model, kT=temps, dt=0.001)
    return state

def test_same_seed_is_fully_reproducible_accept():
    kT, P, acc = attempt_replica_exchange(**_replica_exchange_kwargs(state = _create_md_state(seed=42, temps=torch.Tensor([100, 200])), kT = torch.Tensor([100, 200])))
    assert P == 1 * MetalUnits.pressure
    assert acc == True
    assert kT == torch.Tensor([200, 100])

def test_same_seed_is_fully_reproducible_reject():
    kT, P, acc = attempt_replica_exchange(**_replica_exchange_kwargs(state = _create_md_state(seed=42, temps=torch.Tensor([500, 100])), kT = torch.Tensor([500, 100])))
    assert P == 1 * MetalUnits.pressure
    assert acc == False
    assert kT == torch.Tensor([500, 100])

    
