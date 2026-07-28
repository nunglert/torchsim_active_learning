from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from typing import Any
import json

from torchsim_AL import sampler_spec

import ase.io
import numpy as np
from ase import Atoms
from torchsim_AL.scripts.read_trajectory_to_atoms import _read_trajectory_frames
from remote_handler.core.cluster_handler import ClusterConfig
from remote_handler.core.file_handler import FileSystemHandlerBase
from remote_handler.core.tasks import SingleScriptTask
from remote_handler.workflows.active_learning_base import Sampler

logger = logging.getLogger(__name__)

class TorchSimSampler(Sampler):
    """
    Sampler using torch_sim parallel tempering with MACE models.

    Runs parallel tempering MD (NVT or NPT ensemble) to explore configuration
    space at multiple temperatures. Supports stub mode for testing without
    actual GPU execution.

    Args:
        remote_fs: FileSystemHandlerBase for the machine where sampling runs
        cluster_config: ClusterConfig with partition, conda env, template settings
        sampling_script: Path to torch_sim_sampling.py or compatible script
        founder_path: Path to founder structure file (ASE-readable)
        temperatures: List of replica temperatures in K
        model_paths: List of MACE model identifiers. Use ["foundation"] for
            the MACE-MP foundation model. A single-element list runs a
            standard simulation; multiple entries activate the
            MaceEnsembleModel (shared-body ensemble with uncertainty).
        dt: Integration timestep in ps (default: 0.001)
        n_steps_npt: Number of equilibration steps without swaps (default: 0). These will not be used for selection.
        n_steps_pt: Number of production PT steps (default: 100)
        seed: Random seed (default: 123)
        device: Torch device string (default: "cuda")
        dtype: Torch dtype string (default: "float32")
        remote_savepath: Path on remote machine for calculation files
        loop_time: Interval (seconds) to poll for job completion
    """

    def __init__(
        self,
        remote_fs: FileSystemHandlerBase,
        cluster_config: ClusterConfig,
        sampling_script: os.PathLike,
        founder_path: os.PathLike,
        temperatures: list[float],
        integrator_name: str,
        init_kwargs: dict[str, Any],
        step_kwargs: dict[str, Any],
        target_temperature: Optional[float] = None,
        model_paths: Optional[list[str]] = None,
        dt: float = 0.001,
        external_pressure: Optional[float] = None,
        n_steps_npt: int = 0,
        n_steps_pt: int = 100,
        seed: int = 42,
        device: str = "cuda",
        dtype: str = "float32",
        remote_savepath: Optional[os.PathLike] = None,
        loop_time: float = 30,
        stub: bool = False,
        md_steps_per_exchange: int = 100,
        swap_temperature: bool = True,
        rescale_momenta_on_swap: bool = True,
        rescale_barostat_on_swap: bool = True,
        write_frequency: int=1,
    ):
        super().__init__()
        self.remote_fs = remote_fs
        self.cluster_config = cluster_config
        self.sampling_script = sampling_script
        self.founder_path = founder_path
        self.temperatures = temperatures
        self.target_temperature = target_temperature
        self.write_frequency = write_frequency
        self.model_paths = (
            model_paths if model_paths is not None else ["foundation"]
        )
        self.dt = dt
        self.external_pressure = external_pressure
        self.n_steps_npt = n_steps_npt
        self.n_steps_pt = n_steps_pt
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.remote_savepath = remote_savepath
        self.loop_time = loop_time
        self.stub = stub
        self.md_steps_per_exchange = md_steps_per_exchange
        self.init_kwargs = init_kwargs
        self.step_kwargs = step_kwargs
        self.integrator_name = integrator_name
        self.swap_temperature = swap_temperature
        self.rescale_momenta_on_swap = rescale_momenta_on_swap
        self.rescale_barostat_on_swap = rescale_barostat_on_swap

    def update(
        self, calcdir: Path, sampler_input: list[Path], **kwargs
    ) -> None:
        """
        Prepare sampler for a new iteration.

        Args:
            calcdir: Directory for this AL iteration
            sampler_input: Path or list of paths to model files. The filenames
                are used to populate the ``model_paths`` field in the script
                input JSON. When the AL handler provides finetuned model files
                (from training), those override the initial ``model_paths``
                setting (e.g. ``["foundation"]``).
            **kwargs: Additional parameters
        """
        self.sample_dir = calcdir / "sample"
        self.results_dir = self.sample_dir / "results"
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.model_files = sampler_input if isinstance(sampler_input, list) else [sampler_input]

        is_foundation = self.model_files == ["foundation"]
        if is_foundation:
            current_model_paths = ["foundation"]
        else:
            current_model_paths = [Path(p).name for p in self.model_files]

        logger.info(
            "Validating sampler input for integrator='%s', %d replicas",
            self.integrator_name,
            len(self.temperatures),
        )

        if not self.stub:

            logger.info(
                "Validating sampler input for integrator='%s', %d replicas",
                self.integrator_name,
                len(self.temperatures),
            )

            #Check if model exists.
            if not Path(self.founder_path).exists():
                msg = f"founder_path does not exist locally: {self.founder_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            if not is_foundation:
                missing = [p for p in self.model_files if not Path(p).exists()]
                if missing:
                    msg = f"Model file(s) not found locally, cannot upload to cluster: {missing}"
                    logger.error(msg)
                    raise FileNotFoundError(msg)

            script_input = sampler_spec.validate_spec(
            founder=Path(self.founder_path).name,
            temperatures=self.temperatures,
            dt=self.dt,
            external_pressure=self.external_pressure,
            n_steps_npt=self.n_steps_npt,
            n_steps_pt=self.n_steps_pt,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
            md_steps_per_exchange=self.md_steps_per_exchange,
            write_frequency=self.write_frequency,
            swap_temperature=self.swap_temperature,
            rescale_momenta_on_swap=self.rescale_momenta_on_swap,
            rescale_barostat_on_swap=self.rescale_barostat_on_swap,
            init_kwargs=self.init_kwargs,
            step_kwargs=self.step_kwargs,
            integrator_name=self.integrator_name,
            model_paths=current_model_paths,
            )

            self.model_paths = current_model_paths

            self.script_input_file = self.results_dir / "input_torchsim.json"
            with open(self.script_input_file, "w") as f:
                f.write(json.dumps(script_input, indent=2))

            self.sampling_task = SingleScriptTask(
                local_savepath=self.results_dir,
                input_file=self.script_input_file,
                cluster_config=self.cluster_config,
                remote_fs=self.remote_fs,
                script_file=self.sampling_script,
                remote_savepath=self.remote_savepath,
                jobfile_name="run_torchsim.sh",
                jobname="torchsim_sample",
                logfile="torchsim_sample.log",
                n_processors=1,
                loop_time=self.loop_time,
                collect_only=False,
            )

            self.sampling_task.allocate_data()

            # Copy founder structure to remote
            self.sampling_task.copy_file(self.founder_path)

            # Copy model files to remote (unless using foundation model)
            if not is_foundation:
                for p in self.model_files:
                    self.sampling_task.copy_file(p)

    def _create_stub_outputs(self) -> list[Atoms]:
        """
        Create mock torch_sim sampling outputs for testing.

        Returns:
            List of Atoms objects with mock PT results
        """
        n_replicas = len(self.temperatures)
        logger.info(f"Creating {n_replicas} stub torch_sim samples")

        mock_samples = []
        for i in range(n_replicas):
            symbols = ["Si"] * 32 + ["C"] * 32 + ["Ti"] * 16
            positions = np.random.uniform(0, 12, (len(symbols), 3))

            atoms = Atoms(
                symbols=symbols,
                positions=positions,
                cell=[12, 12, 24],
                pbc=True,
            )

            mock_samples.append(atoms)

        return mock_samples

    def get_sampled_configs(self) -> Path:
        """
        Run torch_sim parallel tempering and collect configurations.

        Returns:
            Path to extxyz file containing sampled configurations
        """
        if self.stub:
            logger.info(
                "Running in stub mode - creating mock torch_sim samples"
            )
            sampled_configs = self._create_stub_outputs()

            output_dir = self.results_dir / "out"
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / "samples.extxyz"
            ase.io.write(output_path, sampled_configs)
        else:
            self.sampling_task.submit()
            self.sampling_task.collect_results()
            # Read all trajectory frames and pass to selector
            results_path = Path(self.sampling_task.local_savepath)
            traj_paths = sorted(
                results_path.glob("trajectory_*.h5md"),
                key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
            )
            if not traj_paths:
                logger.warning(
                    "No trajectory_*.h5md files found in %s, "
                    "falling back to script output",
                    results_path,
                )
                output_path = results_path / "out/samples.extxyz"
            else:
                from torch_sim.trajectory import TorchSimTrajectory

                atoms_list: list[Atoms] = []
                for traj_path in traj_paths:
                    traj = TorchSimTrajectory(str(traj_path), mode="r")
                    atoms = _read_trajectory_frames(traj, self.n_steps_npt, self.write_frequency)
                    traj.close()
                    atoms_list.extend(atoms)

                output_dir = results_path / "out"
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir / "samples.extxyz"
                ase.io.write(output_path, atoms_list)

        logger.info(
            f"Torch_sim sampled configurations written to {output_path}"
        )
        return output_path
