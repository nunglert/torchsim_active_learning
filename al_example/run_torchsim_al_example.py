"""Example: run a torchsim replica exchange active-learning loop.

Run in stub mode first (``STUB = True``) to exercise the wiring/bookkeeping
without touching a cluster.
"""

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
from remote_handler.workflows.selectors import UncertaintySelector, UniformSelector, PipelineSelector

from torchsim_AL.sampler import TorchSimSampler
from torch_sim.units import MetalUnits
import torchsim_AL
import remote_handler

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# --- toggles --------------------------------------------------------------
NUM_ITERATIONS = 4
LOOP_TIME = 10  # seconds between job-completion polls

# --- paths ----------------------------------------------------------------
HOME = Path(os.getcwd())
INPUTS = Path(__file__).resolve().parent / "inputs"
RH_SCRIPTS = (
    Path(remote_handler.__file__).resolve().parent / "scripts"
)
CONFIG = Path(__file__).resolve().parent / "config"
TORCHSIM_AL_SAMPLING_SCRIPT = (
    Path(torchsim_AL.__file__).resolve().parent / "scripts" / "torch_sim_sampling.py"
)
VASP_PP_PATH = "/home/user/vasp_pp_6.4.2"


def main(calchome: str = "/home/user/al_directory"):
    calchome = Path(calchome)

    #Needed if config supplied by json
    #os.environ.setdefault("REMOTE_HANDLER_CONFIG_DIR", str(HOME / "config"))

    # --- file systems -----------------------------------------------------
    local_fs = LocalFileSystem()
    remote_fs_gpu = RemoteFileSystem(remote_alias="datalab")
    remote_fs_cpu = RemoteFileSystem(remote_alias="dirona")

    # GPU cluster that runs jaxrens NS + NeuralIL training.
    gpu_cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "datalab",
            "partition": "GPU-a100,GPU-a40",
            "savepath": "/home/user/dump",
            "conda_env": "torchsim_AL",
            "conda_source": "/home/user/miniconda3",
            "template": "datalab.sh.j2",
        }
    )
    # CPU cluster that runs VASP single-points.
    dft_cluster = ClusterConfig.from_dict(
        {
            "remote_machine": "dirona",
            "partition": "perform,normal",
            "savepath": "/home/user/dump",
            "conda_env": "torchsim_AL",
            "conda_source": "/home/user/miniconda3",
            "template": "dirona.sh.j2",
        }
    )

    # --- SAMPLER: Torchsim parallel tempering replica exchange ------------------
    sampler = TorchSimSampler(
        remote_fs=remote_fs_gpu,
        cluster_config=gpu_cluster,
        remote_savepath=gpu_cluster.savepath,
        sampling_script=TORCHSIM_AL_SAMPLING_SCRIPT,
        loop_time=LOOP_TIME,
        stub=False,
        founder_path=INPUTS / "Cu2S_lowC.extxyz",
        integrator_name="npt_crescale_triclinic",
        temperatures=[
                300.0,
                400.0
            ],
        model_paths=["dummy1", "dummy2"],
        dt=0.001,
        n_steps_prod=100,
        n_steps_equil=5,
        seed=42,
        external_pressure=0, #in bar
        device="cuda",
        dtype="float64",
        md_steps_per_exchange=20,
        step_kwargs = {"tau": 0.1 * MetalUnits.time},
        init_kwargs = {"tau_p": 2.0 * MetalUnits.time, "isothermal_compressibility": 1.23},
        write_frequency = 2
        )

    # --- SELECTOR: Pipeline Selector Uncertianty + QUESTS (remote_handler) ---------------
    selector = PipelineSelector(
        [
            UniformSelector(20)
            UncertaintySelector(n_configs=5),
        ]
    )
    


    # --- EVALUATOR: VASP single-point DFT (remote_handler) ---------------
    evaluator = evaluators.VASPEvaluator(
        singlepoint_script=RH_SCRIPTS / "dft_singlepoint.py",
        remote_fs=remote_fs_cpu,
        cluster_config=dft_cluster,
        n_processors=2,
        num_parallel=5,
        incar_kwargs={
            "xc": "R2SCAN",
            "encut": 500,
            "prec": "Normal",
            "ismear": 0,
            "sigma": 0.05,
            "ibrion": -1,
            "ediff": 1e-5,
            "kspacing": 0.3,
            "lwave": False,
            "lcharg": False,
            "lreal": "Auto",
            "lasph": True,
            "gamma": False,
            "kpar": 1,
            "ncore": 2
        },
        stub=False,
        loop_time=LOOP_TIME,
        remote_savepath=dft_cluster.savepath,
        slurm_header_add="#SBATCH --time=0-20:00:00\n#SBATCH --mem=16GB\nmodule purge\nmodule load vasp/6.4.2_intel\nsource /opt/intel/oneapi/setvars.sh --force\nexport VASP_PP_PATH=" + VASP_PP_PATH,
    )

    # --- TRAINER: NeuralIL force field (remote_handler) ------------------
    trainer = trainers.MaceFinetuningTrainer(
        remote_fs=remote_fs_gpu,
        cluster_config=gpu_cluster,
        training_script=RH_SCRIPTS / "mace" / "mace_finetuning.py",
        config_file=CONFIG / "mace_config.yaml",
        remote_mace_train_runner=Path("/home/bruno.bachmair/miniconda3/envs/torchsim_AL/bin/mace_run_train"),
        foundation_model=INPUTS / "2023-12-10-mace-128-L0_energy_epoch-249.model",
        num_models=3,
        n_processors=3,
        remote_savepath=Path("/home/user/dump"),
        loop_time=LOOP_TIME,
    )
    
    # --- INITIAL SAMPLER: Torchsim parallel tempering replica exchange --- #
    initial_torchsim_sampler = TorchSimSampler(
            remote_fs=remote_fs_gpu,
            cluster_config=gpu_cluster,
            remote_savepath=gpu_cluster.savepath,
            sampling_script=TORCHSIM_AL_SAMPLING_SCRIPT,
            loop_time=LOOP_TIME,
            stub=False,
            founder_path=INPUTS / "Cu2S_lowC.extxyz",
            integrator_name="npt_crescale_triclinic",
            temperatures=[
                300.0,
                400.0
            ],
            model_paths=["foundation"],
            dt=0.001,
            n_steps_prod=20,
            n_steps_equil=0,
            seed=42,
            external_pressure=0, #in bar
            device="cuda",
            dtype="float64",
            md_steps_per_exchange=10,
            step_kwargs = {"tau": 0.1 * MetalUnits.time},
            init_kwargs = {"tau_p": 2.0 * MetalUnits.time, "isothermal_compressibility": 1.23},
            write_frequency = 2
            )

    initial_sample_file = calchome / "initial_samples.extxyz"
    if os.path.exists(initial_sample_file):
        initial_atoms = ase.io.read(initial_sample_file, index=":")
    else:
        initial_torchsim_sampler.update(
            calchome / "initial_samples", [INPUTS / "2023-12-10-mace-128-L0_energy_epoch-249.model"]
        )
        output_path = initial_torchsim_sampler.get_sampled_configs()
        initial_atoms = ase.io.read(output_path, index="::5")
    ase.io.write(initial_sample_file, initial_atoms)

    # --- orchestrate ------------------------------------------------------
    handler = StagedActiveLearningHandler.from_actors(
        sampler=sampler,
        evaluator=evaluator,
        trainer=trainer,
        calchome=str(calchome),
        selector=selector,
        initial_sampler_out=initial_sample_file,
        cycle_entry_point="evaluation",
        num_iterations=NUM_ITERATIONS,
    )
    handler.run()

    print(f"AL run complete: {calchome}")


if __name__ == "__main__":
    main()
