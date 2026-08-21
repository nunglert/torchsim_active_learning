"""Tests for the shared-body MACE ensemble wrapper.

These need mace-torch plus a real model, so they skip cleanly on a machine that
has neither. Rather than shipping a checkpoint with the repo, the fixture pulls
the small MACE-MP foundation model -- the same one ``torch_sim_sampling`` loads
for ``model_paths=["foundation"]`` -- which mace caches under ``~/.cache/mace``
after the first download. No network and no cache -> skip.

The core claim under test: with N identical members, the ensemble mean must
reproduce the single-model result exactly and the uncertainty must collapse to
zero.
"""

import pytest
import torch

pytest.importorskip("mace", reason="mace-torch not installed")

from torchsim_AL.scripts.make_mace import MaceEnsembleModel, _make_model  # noqa: E402


@pytest.fixture(scope="module")
def raw_mace():
    """The small MACE-MP foundation model, downloaded once and cached by mace."""
    from mace.calculators.foundations_models import mace_mp

    try:
        return mace_mp(model="small", device="cpu", return_raw_model=True)
    except Exception as exc:  # no cached copy and no network
        pytest.skip(f"MACE-MP foundation model unavailable: {exc}")


@pytest.fixture
def sim_state(small_structure):
    import torch_sim as ts

    return ts.initialize_state(
        [small_structure], torch.device("cpu"), torch.float64
    )


def test_make_model_single_returns_mace_model(raw_mace):
    from torch_sim.models.mace import MaceModel

    model = _make_model(
        mace=raw_mace,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )
    assert isinstance(model, MaceModel)


def test_make_model_list_returns_ensemble(raw_mace):
    model = _make_model(
        mace=[raw_mace, raw_mace],
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )
    assert isinstance(model, MaceEnsembleModel)
    assert model.n_members == 2


def test_ensemble_requires_at_least_two_members(raw_mace):
    with pytest.raises(ValueError, match="at least 2"):
        MaceEnsembleModel(
            models=[raw_mace],
            device=torch.device("cpu"),
            dtype=torch.float64,
            compute_stress=False,
        )


def test_identical_members_give_zero_uncertainty(raw_mace, sim_state):
    """Two copies of one model must disagree by exactly nothing."""
    ensemble = _make_model(
        mace=[raw_mace, raw_mace],
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )
    results = ensemble(sim_state)

    assert torch.allclose(
        results["energy_uncertainty"],
        torch.zeros_like(results["energy_uncertainty"]),
        atol=1e-8,
    )
    assert torch.allclose(
        results["forces_uncertainty"],
        torch.zeros_like(results["forces_uncertainty"]),
        atol=1e-8,
    )


def test_identical_members_match_single_model(raw_mace, sim_state):
    single = _make_model(
        mace=raw_mace,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )
    ensemble = _make_model(
        mace=[raw_mace, raw_mace],
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )

    single_out = single(sim_state)
    ensemble_out = ensemble(sim_state)

    assert torch.allclose(
        single_out["energy"], ensemble_out["energy"], atol=1e-8
    )
    assert torch.allclose(
        single_out["forces"], ensemble_out["forces"], atol=1e-8
    )


def test_report_counter_resets_each_forward(raw_mace, sim_state):
    """prop_calculators index into the batch via this counter; a stale value
    would silently read the wrong system's uncertainty."""
    ensemble = _make_model(
        mace=[raw_mace, raw_mace],
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=False,
    )
    ensemble(sim_state)
    ensemble._uncertainty_report_counter = 5
    ensemble(sim_state)
    assert ensemble._uncertainty_report_counter == 0