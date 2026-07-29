import copy

import torch
from ase import Atoms
import torch_sim as ts
from torch_sim.integrators.nvt import nvt_langevin_init, nvt_langevin_step
from torch_sim.units import MetalUnits
from torch_sim.models.lennard_jones import LennardJonesModel

from torchsim_AL.scripts.replica_exchange import attempt_replica_exchange

model = LennardJonesModel(
    sigma=3.4, epsilon=0.01, device=torch.device("cpu"), dtype=torch.float64,
    compute_forces=True, compute_stress=True, cutoff=6.0,
)

TEMPS_K = torch.tensor([300.0, 800.0], dtype=torch.float64)
ACCEPT_STATE_SEED, ACCEPT_EXCHANGE_SEED = 0, 0
REJECT_STATE_SEED, REJECT_EXCHANGE_SEED = 2, 0


def _replica_exchange_kwargs(**overrides):
    kwargs = dict(
        i=0,
        j=1,
        rng=torch.Generator(device="cpu").manual_seed(42),
        kT=TEMPS_K * MetalUnits.temperature,
        external_pressure=None,  # NVT: no P*V term
        rescale_momenta=True,
        rescale_barostat=True,
    )
    kwargs.update(overrides)
    return kwargs


def _create_md_state(seed, temps_K=TEMPS_K, n_steps=20):
    small_structure = Atoms(symbols=["Ar"] * 2, positions=[(1, 2, 3), (3, 2, 1)], cell=[10, 10, 10], pbc=True)
    replica_atoms = [copy.deepcopy(small_structure) for _ in range(2)]
    initial_state = ts.initialize_state(replica_atoms, torch.device("cpu"), torch.float64)
    initial_state.rng = seed
    dt = torch.tensor(0.001 * MetalUnits.time, dtype=torch.float64)
    temps = temps_K * MetalUnits.temperature
    state = nvt_langevin_init(state=initial_state, model=model, kT=temps, dt=dt)
    for _ in range(n_steps):
        state = nvt_langevin_step(state=state, model=model, kT=temps, dt=dt, gamma=1.0)
    return state

def test_same_seed_is_fully_reproducible_accept():
    rng = torch.Generator(device="cpu").manual_seed(ACCEPT_EXCHANGE_SEED)
    kT, P, acc = attempt_replica_exchange(
        **_replica_exchange_kwargs(state=_create_md_state(seed=ACCEPT_STATE_SEED), rng=rng)
    )
    assert P is None
    assert acc is True
    assert torch.equal(kT, TEMPS_K.flip(0) * MetalUnits.temperature)

    rng2 = torch.Generator(device="cpu").manual_seed(ACCEPT_EXCHANGE_SEED)
    kT2, P2, acc2 = attempt_replica_exchange(
        **_replica_exchange_kwargs(state=_create_md_state(seed=ACCEPT_STATE_SEED), rng=rng2)
    )
    assert acc2 is True
    assert torch.equal(kT, kT2)


def test_same_seed_is_fully_reproducible_reject():
    rng = torch.Generator(device="cpu").manual_seed(REJECT_EXCHANGE_SEED)
    kT, P, acc = attempt_replica_exchange(
        **_replica_exchange_kwargs(state=_create_md_state(seed=REJECT_STATE_SEED), rng=rng)
    )
    assert P is None
    assert acc is False
    assert torch.equal(kT, TEMPS_K * MetalUnits.temperature)

    rng2 = torch.Generator(device="cpu").manual_seed(REJECT_EXCHANGE_SEED)
    kT2, P2, acc2 = attempt_replica_exchange(
        **_replica_exchange_kwargs(state=_create_md_state(seed=REJECT_STATE_SEED), rng=rng2)
    )
    assert acc2 is False
    assert torch.equal(kT, kT2)