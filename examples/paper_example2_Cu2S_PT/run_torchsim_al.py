import logging
import os
import shutil
import sys
import ase
from pathlib import Path

from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import LocalFileSystem, RemoteFileSystem
from remote_handler.workflows import (
    StagedActiveLearningHandler,
    active_learning_base,
    evaluators,
    trainers,
)
from remote_handler.workflows.active_learning_base import Schedule
from remote_handler.workflows.selectors import UncertaintySelector, QUESTSSelector, PipelineSelector

from torchsim_AL.sampler import TorchSimSampler
from torch_sim.units import MetalUnits
import torchsim_AL
import remote_handler

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- toggles --------------------------------------------------------------
NUM_ITERATIONS = 26
LOOP_TIME = 30  # seconds between job-completion polls

# --- paths ----------------------------------------------------------------
HOME = Path(os.getcwd())
_tmpdir = os.path.expanduser("~/tmp")
os.makedirs(_tmpdir, exist_ok=True)
os.environ["TMPDIR"] = _tmpdir
INPUTS = Path(__file__).resolve().parent / "inputs"

CONFIG = Path(__file__).resolve().parent / "config"
TORCHSIM_AL_SAMPLING_SCRIPT = (
    Path(torchsim_AL.__file__).resolve().parent / "scripts" / "torch_sim_sampling.py"
)
VASP_PP_PATH = "/home/bbachmair/vasp_pp_6.4.2"

# You'll need to add this structure file next to Cu2S_monoclinic.extxyz.
FOUNDER_MONOCLINIC = INPUTS / "Cu2S_monoclinic.extxyz"
FOUNDER_HEXAGONAL = INPUTS / "Cu2S_hexagonal_3_3_2.extxyz"

# Two MACE configs: rename your current config/mace_config.yaml to the
# "early" one, and add a second yaml for the later iterations.
MACE_CONFIG_EARLY = CONFIG / "mace_config_early.yaml"
MACE_CONFIG_LATE = CONFIG / "mace_config_late.yaml"


# ---------------------------------------------------------------------------
# Wrappers: apply Schedule kwargs as attribute overrides before update()
# ---------------------------------------------------------------------------
#
# Workflows before commit b2f9ad55 required this wrapping of the AL stages in 
# case schedules were to be supplied. This was fixed in b2f9ad55.
#
class ScheduledTorchSimSampler(TorchSimSampler):
    def update(self, calcdir, sampler_input, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().update(calcdir, sampler_input)


class ScheduledVASPEvaluator(evaluators.VASPEvaluator):
    def update(self, calcdir, evaluator_input, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().update(calcdir, evaluator_input)


class ScheduledMaceFinetuningTrainer(trainers.MaceFinetuningTrainer):
    def update(self, calcdir, trainer_input, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().update(calcdir, trainer_input)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def _build_slurm_header(mem_gb: int) -> str:
    return (
        "#SBATCH --time=0-20:00:00\n"
        f"#SBATCH --mem={mem_gb}GB\n"
        "module purge\n"
        "module load vasp/6.4.2_intel\n"
        "source /opt/intel/oneapi/setvars.sh --force\n"
        f"export VASP_PP_PATH={VASP_PP_PATH}"
    )


class StructureSamplerSchedule(Schedule):
    """Switches founder structure + n_steps_prod across the AL run.

    it 0     : sampler stage doesn't actually run (cycle_entry_point is
               "evaluation"), so these values are never consumed -- but
               get_kwargs(0) still has to return something with the same
               keys as every other iteration for Schedule.check() to pass.
    it 1-10  : monoclinic, 25000 steps
    it 11-12 : monoclinic, 50000 steps
    it 13-22 : hexagonal,  25000 steps
    it 23-24 : hexagonal,  50000 steps
    it 25-26 : hexagonal, 200000 steps
    """

    def get_kwargs(self, iteration_idx: int) -> dict:
        if iteration_idx == 0:
            founder, len, interv, temp = FOUNDER_MONOCLINIC, 25000, 2, [320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,]
        elif iteration_idx <= 10:
            founder, len, interv, temp = FOUNDER_MONOCLINIC, 25000, 2, [320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,]
        elif iteration_idx <= 12:
            founder, len, interv, temp = FOUNDER_MONOCLINIC, 50000, 4, [320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,]
        elif iteration_idx <= 22:
            founder, len, interv, temp = FOUNDER_HEXAGONAL, 25000, 2, [320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,]
        elif iteration_idx <= 24:
            founder, len, interv, temp = FOUNDER_HEXAGONAL, 50000, 4, [320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,]
        else:
            founder, len, interv, temp = FOUNDER_HEXAGONAL, 200000, 8, [600.0, 650.0, 700.0,]

        return {
            "founder_path": str(founder),
            "n_steps_prod": len,
            "write_frequency": interv,
            "temperatures": temp,
        }


class EvaluatorMemSchedule(Schedule):
    """Only the SLURM header changes: 38GB while evaluating monoclinic
    samples (it 0-12, matching the sampler schedule above), 24GB once the
    hexagonal supercell takes over (it 13 onward)."""

    def get_kwargs(self, iteration_idx: int) -> dict:
        mem_gb = 38 if iteration_idx <= 12 else 24
        return {"slurm_header_add": _build_slurm_header(mem_gb)}


class TrainerConfigSchedule(Schedule):
    """Early-iteration MACE config through it 4, later config from it 5
    onward."""

    def get_kwargs(self, iteration_idx: int) -> dict:
        config_file = MACE_CONFIG_EARLY if iteration_idx < 5 else MACE_CONFIG_LATE
        return {"config_file": str(config_file)}


def main(calchome: str = "/home/bruno/Masterarbeit/active_learning/monoclinic_run_principled"):
    calchome = Path(calchome)

    # --- file systems -----------------------------------------------------
    local_fs = LocalFileSystem()
    remote_fs_gpu = RemoteFileSystem(remote_alias="datalab", local_temp_path=_tmpdir)
    remote_fs_cpu = RemoteFileSystem(remote_alias="dirona", local_temp_path=_tmpdir)

    # GPU cluster that runs torchsim_sampling.
    gpu_cluster_sample = ClusterConfig.from_dict(
        {
            "remote_machine": "datalab",
            "partition": "GPU-a100,GPU-a40,GPU-rtx6000",
            "savepath": "/home/bruno.bachmair/Masterarbeit/dump",
            "conda_env": "torchsim_AL",
            "conda_source": "/home/bruno.bachmair/miniconda3",
            "template": "datalab.sh.j2",
            "resilient_connection": True,
            "connection_retry_interval": 60,
            "connection_max_wait": 600,
            "connection_command_timeout": 60,
            "slurm_header_add": "#SBATCH --time=0-12:00:00"
        }
    )
    # GPU cluster that runs MACE training.
    gpu_cluster_train = ClusterConfig.from_dict(
        {
            "remote_machine": "datalab",
            "partition": "GPU-a100,GPU-a100s,GPU-a40,GPU-l40s,GPU-rtx6000",
            "savepath": "/home/bruno.bachmair/Masterarbeit/dump",
            "conda_env": "torchsim_AL",
            "conda_source": "/home/bruno.bachmair/miniconda3",
            "template": "datalab.sh.j2",
            "resilient_connection": True,
            "connection_retry_interval": 60,
            "connection_max_wait": 600,
            "connection_command_timeout": 60,
            "slurm_header_add": "#SBATCH --time=0-03:00:00"
        }
    )
    # CPU cluster that runs VASP single-points.
    dft_cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "dirona",
            "partition": "perform,normal",
            "savepath": "/home/bbachmair/Masterarbeit/dump",
            "conda_env": "torchsim_AL",
            "conda_source": "/home/bbachmair/packages/miniconda3",
            "template": "dirona.sh.j2",
            "resilient_connection": True,
            "connection_retry_interval": 60,
            "connection_max_wait": 600,
            "connection_command_timeout": 60,
        }
    )

    # --- SAMPLER: Torchsim parallel tempering replica exchange ------------------
    sampler = ScheduledTorchSimSampler(
        remote_fs=remote_fs_gpu,
        cluster_config=gpu_cluster_sample,
        remote_savepath=gpu_cluster_sample.savepath,
        sampling_script=TORCHSIM_AL_SAMPLING_SCRIPT,
        loop_time=LOOP_TIME,
        stub=False,
        founder_path=FOUNDER_MONOCLINIC,
        integrator_name="npt_crescale_triclinic",
        temperatures=[
                320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0,
                480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,
            ],
        model_paths=["dummy1", "dummy2"],
        dt=0.002,
        n_steps_prod=25000,
        n_steps_equil=0,
        seed=42,
        external_pressure=0,  # in bar
        device="cuda",
        dtype="float64",
        md_steps_per_exchange=250,
        step_kwargs={"tau": 0.2 * MetalUnits.time},
        init_kwargs={"tau_p": 0.5 * MetalUnits.time, "isothermal_compressibility": 1.23},
        write_frequency=2,
        )

    # --- SELECTOR: Pipeline Selector Uncertainty + QUESTS (remote_handler) ---------------
    selector = PipelineSelector(
        [
            UncertaintySelector(n_configs=600),
            QUESTSSelector(n_configs=50, method="msc"),
        ]
    )

    # --- EVALUATOR: VASP single-point DFT (remote_handler) ---------------
    evaluator = ScheduledVASPEvaluator(
        remote_fs=remote_fs_cpu,
        cluster_config=dft_cluster,
        n_processors=32,
        num_parallel=25,
        incar_kwargs={
            "xc": "R2SCAN",
            "encut": 500,
            "prec": "Normal",
            "ismear": 0,
            "isym": 0,
            "symprec": 0.00001,
            "sigma": 0.05,
            "ibrion": -1,
            "ediff": 1e-5,
            "kspacing": 0.3,
            "lwave": False,
            "lcharg": False,
            "lreal": "Auto",
            "lasph": True,
            "kgamma": True,
            "kpar": 4,
            "ncore": 8,
        },
        stub=False,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
        slurm_header_add=_build_slurm_header(38),  # overridden per-iteration by the schedule
    )

    # --- TRAINER: MACE force field (remote_handler) ------------------
    trainer = ScheduledMaceFinetuningTrainer(
        remote_fs=remote_fs_gpu,
        cluster_config=gpu_cluster_train,
        config_file=MACE_CONFIG_EARLY,  # overridden per-iteration by the schedule
        remote_mace_train_runner=Path("/home/bruno.bachmair/miniconda3/envs/torchsim_AL/bin/mace_run_train"),
        foundation_model=INPUTS / "mace-mp-0b3-medium.model",
        num_models=4,
        n_processors=4,
        remote_savepath=gpu_cluster_train.savepath,
        loop_time=LOOP_TIME,
        convert_to_jax=False,
    )

    # --- INITIAL SAMPLER: Torchsim parallel tempering replica exchange --- #
    initial_torchsim_sampler = TorchSimSampler(
            remote_fs=remote_fs_gpu,
            cluster_config=gpu_cluster_sample,
            remote_savepath=gpu_cluster_sample.savepath,
            sampling_script=TORCHSIM_AL_SAMPLING_SCRIPT,
            loop_time=LOOP_TIME,
            stub=False,
            founder_path=INPUTS / "Cu2S_monoclinic.extxyz",
            integrator_name="npt_crescale_triclinic",
            temperatures=[
                320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0,
                480.0, 500.0, 530.0, 560.0, 600.0, 650.0, 700.0,
            ],
            model_paths=["foundation"],
            dt=0.002,
            n_steps_prod=20,
            n_steps_equil=0,
            seed=42,
            external_pressure=0,  # in bar
            device="cuda",
            dtype="float32",
            md_steps_per_exchange=10,
            step_kwargs={"tau": 0.2 * MetalUnits.time},
            init_kwargs={"tau_p": 0.5 * MetalUnits.time, "isothermal_compressibility": 1.23},
            write_frequency=2,
            )

    initial_sample_file = calchome / "initial_samples.extxyz"
    if os.path.exists(initial_sample_file):
        initial_atoms = ase.io.read(initial_sample_file, index=":")
    else:
        initial_torchsim_sampler.update(
            calchome / "initial_samples", [INPUTS / "mace-mp-0b3-medium.model"]
        )
        output_path = initial_torchsim_sampler.get_sampled_configs()
        initial_atoms = ase.io.read(output_path, index="::9")
    ase.io.write(initial_sample_file, initial_atoms)

    # --- SCHEDULES ----------------------------------------------------
    sampler_schedule = StructureSamplerSchedule(NUM_ITERATIONS)
    evaluator_schedule = EvaluatorMemSchedule(NUM_ITERATIONS)
    trainer_schedule = TrainerConfigSchedule(NUM_ITERATIONS)

    # Sanity-check the schedules before touching a cluster.
    #sampler_schedule.visualize()
    evaluator_schedule.visualize()
    trainer_schedule.visualize()

    # --- orchestrate ------------------------------------------------------
    handler = StagedActiveLearningHandler.from_actors(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        selector=selector,
        initial_sampler_out=initial_sample_file,
        cycle_entry_point="evaluation",
        cycle_exit_point="training",
        num_iterations=NUM_ITERATIONS,
        sampler_schedule=sampler_schedule,
        evaluator_schedule=evaluator_schedule,
        trainer_schedule=trainer_schedule,
    )
    handler.run()

    print(f"AL run complete: {calchome}")


if __name__ == "__main__":
    main()
