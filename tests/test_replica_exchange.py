import pytest
from torchsim_AL.scripts.replica_exchange import attempt_replica_exchange
import torch
import copy
from ase import Atoms
import torch_sim as ts
from torch_sim.integrators.nvt import nvt_langevin_init
from torch_sim.units import MetalUnits

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

@pytest.fixture
def lj_model():
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(
        sigma=3.4, epsilon=0.01, device=torch.device("cpu"), dtype=torch.float64,
        compute_forces=True, compute_stress=True, cutoff=6.0,
    )

@pytest.fixture
def _create_md_state(seed, temps):
    small_structure = Atoms(symbols=["Ar"], positions=[1, 2, 3], cell=[10, 10, 10], pbc=True)
    replica_atoms = [copy.deepcopy(small_structure) for _ in range(2)]
    initial_state = ts.initialize_state(replica_atoms, torch.device("cpu"), torch.dtype("float64"))
    initial_state.rng = seed
    state = nvt_langevin_init(state=initial_state, model=lj_model, kT=temps, dt=0.001)
    return state

def test_same_seed_is_fully_reproducible_accept():
    kT, P, acc = attempt_replica_exchange(**_replica_exchange_kwargs(state = _create_md_state(seed=42, temps=torch.Tensor([100, 200])), kT = torch.Tensor([100, 200])))
    assert P == 1 * MetalUnits.pressure
    assert acc == True
    assert kT == torch.Tensor([200, 100])

def test_same_seed_is_fully_reproducible_reject():
    kT, P, acc = attempt_replica_exchange(**_replica_exchange_kwargs(state = _create_md_state(seed=42, temps=torch.Tensor([200, 100])), kT = torch.Tensor([200, 100])))
    assert P == 1 * MetalUnits.pressure
    assert acc == False
    assert kT == torch.Tensor([200, 100])

    
