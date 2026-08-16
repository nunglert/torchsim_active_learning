import json

import pytest
from ase.io import read

from torchsim_AL.sampler import TorchSimSampler

from conftest import FRAMES_AFTER_BURN_IN


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


def test_update_creates_directories(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    assert sampler.sample_dir == tmp_path / "0" / "sample"
    assert sampler.results_dir == tmp_path / "0" / "sample" / "results"
    assert sampler.sample_dir.is_dir()
    assert sampler.results_dir.is_dir()


def test_update_is_idempotent_across_iterations(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    assert len(recording_task.instances) == 2


def test_update_writes_valid_input_json(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    assert sampler.script_input_file == sampler.results_dir / "input_torchsim.json"
    spec = json.loads(sampler.script_input_file.read_text())

    assert spec["integrator_name"] == "nvt_langevin"
    assert spec["temperatures"] == [300.0, 400.0, 500.0]
    assert spec["step_kwargs"] == {"gamma": 1.0}
    assert spec["model_paths"] == ["foundation"]
    assert spec["device"] == "cpu"
    assert spec["founder"] == "founder.extxyz"


def test_update_seed_increments_with_iteration_number(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(**sampler_kwargs)

    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    assert json.loads(sampler.script_input_file.read_text())["seed"] == 42

    sampler.update(calcdir=tmp_path / "7", sampler_input=["foundation"])
    assert json.loads(sampler.script_input_file.read_text())["seed"] == 49


def test_update_non_numeric_calcdir_falls_back_to_iteration_zero(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "not_a_number", sampler_input=["foundation"])
    assert json.loads(sampler.script_input_file.read_text())["seed"] == 42


def test_update_explicit_seed_kwarg_overrides_iteration(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "7", sampler_input=["foundation"], seed=123)
    assert json.loads(sampler.script_input_file.read_text())["seed"] == 123


def test_update_sampler_input_overrides_model_paths(
    tmp_path, sampler_kwargs, recording_task, fake_model_files
):
    sampler = TorchSimSampler(**{**sampler_kwargs, "model_paths": ["foundation"]})
    sampler.update(calcdir=tmp_path / "0", sampler_input=fake_model_files)

    spec = json.loads(sampler.script_input_file.read_text())
    # Basenames only, for the same reason as founder.
    assert spec["model_paths"] == ["model_a.model", "model_b.model"]
    assert sampler.model_paths == ["model_a.model", "model_b.model"]


def test_update_accepts_bare_path_as_sampler_input(
    tmp_path, sampler_kwargs, recording_task, fake_model_files
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=fake_model_files[0])

    spec = json.loads(sampler.script_input_file.read_text())
    assert spec["model_paths"] == ["model_a.model"]


def test_update_copies_founder_and_models_to_task(
    tmp_path, sampler_kwargs, recording_task, fake_model_files
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=fake_model_files)

    task = recording_task.instances[-1]
    assert task.allocated == 1
    assert task.copied_files == [sampler_kwargs["founder_path"], *fake_model_files]


def test_update_foundation_model_copies_only_founder(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    task = recording_task.instances[-1]
    assert task.copied_files == [sampler_kwargs["founder_path"]]


def test_update_wires_task_to_results_dir(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    task = recording_task.instances[-1]
    assert task.kwargs["local_savepath"] == sampler.results_dir
    assert task.kwargs["input_file"] == sampler.script_input_file
    assert task.kwargs["collect_only"] is False


def test_update_missing_founder_raises(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(
        **{**sampler_kwargs, "founder_path": tmp_path / "nope.extxyz"}
    )
    with pytest.raises(FileNotFoundError, match="founder_path"):
        sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    assert recording_task.instances == []


def test_update_missing_model_file_raises(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(**sampler_kwargs)
    with pytest.raises(FileNotFoundError, match="Model file"):
        sampler.update(
            calcdir=tmp_path / "0", sampler_input=[tmp_path / "ghost.model"]
        )
    assert recording_task.instances == []


def test_update_invalid_config_raises_before_touching_cluster(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(
        **{**sampler_kwargs, "temperatures": [500.0, 300.0, 400.0]}
    )
    with pytest.raises(ValueError, match="increasing"):
        sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    assert recording_task.instances == []
    assert not (tmp_path / "0" / "sample" / "results" / "input_torchsim.json").exists()


def test_update_stub_mode_skips_validation_and_task(
    tmp_path, sampler_kwargs, recording_task
):
    sampler = TorchSimSampler(
        **{**sampler_kwargs, "stub": True, "temperatures": [500.0, 300.0]}
    )
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    assert sampler.results_dir.is_dir()
    assert recording_task.instances == []
    assert not hasattr(sampler, "script_input_file")


# ---------------------------------------------------------------------------
# get_sampled_configs()
# ---------------------------------------------------------------------------


def test_get_sampled_configs_stub_mode(tmp_path, sampler_kwargs, recording_task):
    sampler = TorchSimSampler(**{**sampler_kwargs, "stub": True})
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    output_path = sampler.get_sampled_configs()

    assert output_path == sampler.results_dir / "out" / "samples.extxyz"
    assert output_path.exists()
    sampled = read(output_path, index=":")
    assert len(sampled) == len(sampler_kwargs["temperatures"])


def test_get_sampled_configs_collects_all_replica_trajectories(
    tmp_path, sampler_kwargs, recording_task, seeded_results_dir
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    trajectories = seeded_results_dir(sampler.results_dir)

    output_path = sampler.get_sampled_configs()

    assert output_path == sampler.results_dir / "out" / "samples.extxyz"
    sampled = read(output_path, index=":")
    assert len(sampled) == len(trajectories) * FRAMES_AFTER_BURN_IN


def test_get_sampled_configs_submits_and_collects(
    tmp_path, sampler_kwargs, recording_task, seeded_results_dir
):
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    seeded_results_dir(sampler.results_dir)

    sampler.get_sampled_configs()

    task = recording_task.instances[-1]
    assert task.submitted == 1
    assert task.collected == 1


def test_get_sampled_configs_preserves_per_frame_properties(
    tmp_path, sampler_kwargs, recording_task, seeded_results_dir
):
    """Uncertainty/energy props must survive the h5md -> extxyz hand-off,
    since the selector downstream keys off them."""
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    seeded_results_dir(sampler.results_dir)

    sampled = read(sampler.get_sampled_configs(), index=":")
    assert all("potential_energy" in atoms.info for atoms in sampled)


def test_get_sampled_configs_sorts_trajectories_numerically(
    tmp_path, sampler_kwargs, recording_task, seeded_results_dir, monkeypatch
):
    """trajectory_10 must come after trajectory_9, not after trajectory_1."""
    import torchsim_AL.sampler as sampler_module

    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])
    seeded_results_dir(sampler.results_dir, n_copies=11)

    seen = []

    def _spy(traj, n_steps_equil, write_frequency):
        seen.append(traj.filename)
        return []

    monkeypatch.setattr(sampler_module, "_read_trajectory_frames", _spy)
    sampler.get_sampled_configs()

    indices = [int(str(name).rsplit("_", 1)[-1].split(".")[0]) for name in seen]
    assert indices == list(range(11))


def test_get_sampled_configs_without_trajectories_returns_missing_path(
    tmp_path, sampler_kwargs, recording_task, caplog
):
    """Documents the current fallback: when collect_results() brings nothing
    back, the sampler warns and hands the selector a path to a file that no
    longer gets written by anything."""
    sampler = TorchSimSampler(**sampler_kwargs)
    sampler.update(calcdir=tmp_path / "0", sampler_input=["foundation"])

    with caplog.at_level("WARNING", logger="torchsim_AL.sampler"):
        output_path = sampler.get_sampled_configs()

    assert any("No trajectory" in r.message for r in caplog.records)
    assert not output_path.exists()