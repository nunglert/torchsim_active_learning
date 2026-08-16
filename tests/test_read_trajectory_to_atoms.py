"""Tests for _read_trajectory_frames: burn-in slicing and property attachment.

Frame k in the trajectory corresponds to MD step k * write_frequency, because
``parallel_tempering`` writes an initial frame at step 0 before the first
integration step.
"""

import pytest
from torch_sim import TorchSimTrajectory

from torchsim_AL.scripts.read_trajectory_to_atoms import _read_trajectory_frames

from conftest import PT_N_FRAMES, PT_N_STEPS_EQUIL, PT_N_STEPS_PROD, PT_WRITE_FREQUENCY, FRAMES_AFTER_BURN_IN


@pytest.fixture
def trajectory(run_pt, tmp_path):
    """One finished replica trajectory, opened read-only."""
    _, filenames = run_pt(tmp_path)
    traj = TorchSimTrajectory(str(filenames[0]), mode="r")
    yield traj
    traj.close()


def test_writes_expected_number_of_frames(trajectory):
    assert len(trajectory) == PT_N_FRAMES
    assert int(trajectory.get_steps("positions")[0]) == 0
    assert int(trajectory.get_steps("positions")[-1]) == (
        PT_N_STEPS_EQUIL + PT_N_STEPS_PROD
    )


def test_burn_in_drops_equilibration_frames(trajectory):
    frames = _read_trajectory_frames(trajectory, PT_N_STEPS_EQUIL, PT_WRITE_FREQUENCY)
    assert len(frames) == FRAMES_AFTER_BURN_IN


def test_zero_burn_in_returns_every_frame(trajectory):
    frames = _read_trajectory_frames(trajectory, 0, PT_WRITE_FREQUENCY)
    assert len(frames) == PT_N_FRAMES


def test_props_are_attached_to_frames(trajectory):
    frames = _read_trajectory_frames(trajectory, PT_N_STEPS_EQUIL, PT_WRITE_FREQUENCY)
    for atoms in frames:
        # Scalars from prop_calculators land in .info
        assert "potential_energy" in atoms.info
        assert "kinetic_energy" in atoms.info
        assert isinstance(atoms.info["potential_energy"], float)
    # Structural data survives the round trip
    assert len(frames[0]) == 2
    assert frames[0].get_chemical_symbols() == ["Ar", "Ar"]


def test_equil_longer_than_trajectory_warns_and_returns_nothing(trajectory, caplog):
    logger_name = "torchsim_AL.scripts.read_trajectory_to_atoms"
    with caplog.at_level("WARNING", logger=logger_name):
        frames = _read_trajectory_frames(trajectory, 10_000, PT_WRITE_FREQUENCY)
    assert frames == []
    assert any("no production frames" in r.message for r in caplog.records)

def test_burn_in_excludes_frame_at_exactly_n_steps_equil(trajectory):
    frames = _read_trajectory_frames(trajectory, PT_N_STEPS_EQUIL, PT_WRITE_FREQUENCY)
    steps = trajectory.get_steps("positions")
    first_returned_step = int(steps[PT_N_FRAMES - len(frames)])
    assert first_returned_step > PT_N_STEPS_EQUIL