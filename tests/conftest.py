"""Shared fixtures for the torchsim_AL test suite.

Everything here is CPU-only and MACE-free: the Lennard-Jones model stands in
for MACE wherever a torch_sim ``ModelInterface`` is all that's needed, so the
bulk of the suite runs without a GPU, without a cluster, and without the
32 MB model files.
"""

from __future__ import annotations

import copy
import shutil

import pytest
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import write

from torchsim_AL.scripts.parallel_tempering import parallel_tempering
from torchsim_AL.scripts.uncertainty_props import _build_prop_calculators

# Reference PT settings used by the trajectory-producing fixtures. Frames land
# at steps 0, 4, 8, ..., 40 -> 11 frames per replica.
PT_N_STEPS_EQUIL = 10
PT_N_STEPS_PROD = 30
PT_WRITE_FREQUENCY = 4
PT_N_FRAMES = 11
FRAMES_AFTER_BURN_IN = PT_N_FRAMES - (PT_N_STEPS_EQUIL // PT_WRITE_FREQUENCY + 1)


@pytest.fixture
def lj_model():
    """Cheap CPU model with forces + stress, so NVT and NPT both work."""
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(
        sigma=3.4,
        epsilon=0.01,
        device=torch.device("cpu"),
        dtype=torch.float64,
        compute_forces=True,
        compute_stress=True,
        cutoff=6.0,
    )


@pytest.fixture
def small_structure():
    return Atoms(
        symbols=["Ar"] * 2,
        positions=[(1, 2, 3), (3, 2, 1)],
        cell=[10, 10, 10],
        pbc=True,
    )


@pytest.fixture
def replica_atoms(small_structure):
    return [copy.deepcopy(small_structure) for _ in range(3)]


@pytest.fixture
def run_pt(replica_atoms, lj_model):
    """Run a short parallel-tempering job and return (result, trajectory paths).

    Call as ``run_pt(some_dir, **overrides)``. The trajectory files are written
    into ``some_dir`` as ``trajectory_{i}.h5md``, matching the layout
    ``TorchSimSampler.get_sampled_configs`` expects to collect back.
    """

    def _run(out_dir, **overrides):
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
            n_steps_equil=PT_N_STEPS_EQUIL,
            n_steps_prod=PT_N_STEPS_PROD,
            md_steps_per_exchange=5,
            seed=42,
        )
        kwargs.update(overrides)

        n_replicas = len(kwargs["replica_atoms"])
        filenames = [out_dir / f"trajectory_{i}.h5md" for i in range(n_replicas)]
        reporter_args = dict(
            filenames=filenames,
            state_frequency=PT_WRITE_FREQUENCY,
            prop_calculators=_build_prop_calculators(
                kwargs["model"], frequency=PT_WRITE_FREQUENCY
            ),
        )
        return parallel_tempering(**kwargs, reporter_args=reporter_args), filenames

    return _run


@pytest.fixture
def founder_file(tmp_path):
    """A small ASE-readable starting structure on disk."""
    path = tmp_path / "founder.extxyz"
    write(path, bulk("Si", "diamond", a=5.43, cubic=True))
    return path


@pytest.fixture
def fake_model_files(tmp_path):
    """Two files that merely need to *exist* for update()'s presence checks."""
    paths = []
    for name in ("model_a.model", "model_b.model"):
        p = tmp_path / "models" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"not-a-real-mace-model")
        paths.append(p)
    return paths


class RecordingTask:
    """Stand-in for ``remote_handler``'s SingleScriptTask.

    Records the constructor kwargs and every call, so tests can assert on what
    the sampler *asked the cluster layer to do* without a cluster. Deliberately
    not a MagicMock: an unexpected attribute access should fail loudly rather
    than silently auto-create.
    """

    instances: list["RecordingTask"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.local_savepath = kwargs["local_savepath"]
        self.allocated = 0
        self.copied_files = []
        self.submitted = 0
        self.collected = 0
        RecordingTask.instances.append(self)

    def allocate_data(self):
        self.allocated += 1

    def copy_file(self, path):
        self.copied_files.append(path)

    def submit(self):
        self.submitted += 1

    def collect_results(self):
        self.collected += 1


@pytest.fixture
def recording_task(monkeypatch):
    """Replace SingleScriptTask in the sampler module with RecordingTask.

    Yields the class; ``recording_task.instances`` holds every task the sampler
    built during the test.
    """
    import torchsim_AL.sampler as sampler_module

    RecordingTask.instances = []
    monkeypatch.setattr(sampler_module, "SingleScriptTask", RecordingTask)
    yield RecordingTask
    RecordingTask.instances = []


@pytest.fixture
def sampler_kwargs(founder_file):
    """Baseline TorchSimSampler kwargs; override per test."""
    return dict(
        remote_fs=object(),
        cluster_config=object(),
        sampling_script="torch_sim_sampling.py",
        founder_path=founder_file,
        temperatures=[300.0, 400.0, 500.0],
        integrator_name="nvt_langevin",
        init_kwargs={},
        step_kwargs={"gamma": 1.0},
        n_steps_equil=PT_N_STEPS_EQUIL,
        n_steps_prod=PT_N_STEPS_PROD,
        write_frequency=PT_WRITE_FREQUENCY,
        md_steps_per_exchange=5,
        seed=42,
        device="cpu",
        dtype="float64",
    )


@pytest.fixture
def seeded_results_dir(run_pt):
    """Factory: fill a sampler results dir with real trajectory_*.h5md files.

    Used by the get_sampled_configs() tests, which need genuine h5md files that
    ``_read_trajectory_frames`` can actually open.
    """

    def _seed(results_dir, n_copies=None):
        results_dir.mkdir(parents=True, exist_ok=True)
        _, filenames = run_pt(results_dir)
        if n_copies is not None:
            # Duplicate replica 0's trajectory under higher indices. Cheap way
            # to get a double-digit index for the numeric-sort test.
            for i in range(len(filenames), n_copies):
                extra = results_dir / f"trajectory_{i}.h5md"
                shutil.copyfile(filenames[0], extra)
                filenames.append(extra)
        return filenames

    return _seed