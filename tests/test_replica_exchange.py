import pytest
from torchsim_AL.scripts import replica_exchange
import torch
import copy
from ase import Atoms
import torch_sim as ts
from torch_sim.units import MetalUnits

@pytest.fixture
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
def _create_state(seed):
    small_structure = Atoms(symbols=["Ar"], positions=[1, 2, 3], cell=[10, 10, 10], pbc=True)
    replica_atoms = [copy.deepcopy(small_structure) for _ in range(3)]
    initial_state = ts.initialize_state(replica_atoms, torch.device("cpu"), torch.dtype("float64"))
    initial_state.rng = seed


def test_same_seed_is_fully_reproducible_accept():
    kT, P, acc = replica_exchange(**_replica_exchange_kwargs())
    assert P == 1 * MetalUnits.pressure
    assert acc == True
    assert kT == torch.Tensor([200, 100])

def test_same_seed_is_fully_reproducible_reject():
    kT, P, acc = replica_exchange(**_replica_exchange_kwargs(kT = torch.Tensor([200, 100])))
    assert P == 1 * MetalUnits.pressure
    assert acc == False
    assert kT == torch.Tensor([200, 100])

    
