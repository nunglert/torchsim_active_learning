"""The ``mace_eval`` processor -- score configs with torch MACE committees.

A downstream processor for ``remote_handler``'s trajectory-processing framework
(mirroring ``jaxrens_al``'s ``committee_eval``). On the cluster it loads every
member of one or more named committees of torch MACE models (the ``.pkl`` /
``.model`` files a ``MaceFinetuningTrainer`` writes) through
``mace.calculators.MACECalculator`` and annotates each frame with the
committee's predictions. Pickled MACE models carry CUDA-compiled e3nn code and
cannot be unpickled without a GPU, so this runs on a GPU node rather than
host-side.

Only scalars land in ``atoms.info`` by default, so scoring a pool with many
committees stays small on disk. Per committee ``<name>``:

* ``<name>_energy_m<j>`` -- member ``j``'s total energy (eV);
* ``<name>_energy`` / ``<name>_energy_std`` -- committee mean / std;
* ``<name>_forces_std`` -- mean per-atom force std across members (eV/Å);

and, when the frame carries reference forces under ``ref_forces_key``:

* ``<name>_forces_sse_m<j>`` -- member ``j``'s summed squared force error;
* ``<name>_forces_sse`` -- the committee-mean force's summed squared error.

A force RMSE over a set of frames is then ``sqrt(sum(sse) / sum(3 * n_atoms))``.
With ``write_forces`` the committee-mean forces are also written to
``arrays["<name>_forces"]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.base import Processor
from remote_handler.processing.registry import register
from remote_handler.scripts.input_models import ScriptInput


# --- arg schema ------------------------------------------------------------
@dataclass
class MaceEvalArgs(ScriptInput):
    """Args for the ``mace_eval`` processor.

    ``committees`` maps a committee name (the key prefix) to the staged model
    paths of its members on the cluster.
    """

    committees: dict = field(default_factory=dict)
    ref_forces_key: str = "REF_forces"
    device: str = "cuda"
    default_dtype: str = "float64"
    write_forces: bool = False

    def validate(self, root: Optional[Path] = None) -> None:
        if not self.committees:
            raise ScriptInputError("mace_eval: 'committees' must not be empty")
        for name, members in self.committees.items():
            if not name or not isinstance(name, str):
                raise ScriptInputError(
                    "mace_eval: committee names must be non-empty strings"
                )
            if not members or not all(
                isinstance(m, str) and m for m in members
            ):
                raise ScriptInputError(
                    f"mace_eval: committee {name!r} needs at least one "
                    "non-empty model path"
                )
        if self.default_dtype not in ("float32", "float64"):
            raise ScriptInputError(
                "mace_eval: 'default_dtype' must be 'float32' or 'float64'"
            )


# --- cluster-side compute --------------------------------------------------
def _load_calculator(model_path, device, default_dtype):
    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=model_path, device=device, default_dtype=default_dtype
    )


def _predict(calc, atoms_list):
    energies, forces = [], []
    for atoms in atoms_list:
        at = atoms.copy()
        at.calc = calc
        energies.append(float(at.get_potential_energy()))
        forces.append(at.get_forces().copy())
    return energies, forces


def run(atoms_list, args, calcpath=None):
    """Annotate each structure with every committee's predictions.

    Members are loaded one at a time and scored over the whole shard, so a
    committee costs one model load per member, not one per frame.
    """
    import gc

    import numpy as np

    device = args.get("device", "cuda")
    default_dtype = args.get("default_dtype", "float64")
    ref_key = args.get("ref_forces_key", "REF_forces")
    write_forces = bool(args.get("write_forces", False))

    for name, members in args["committees"].items():
        member_energies, member_forces = [], []
        for path in members:
            calc = _load_calculator(path, device, default_dtype)
            energies, forces = _predict(calc, atoms_list)
            member_energies.append(energies)
            member_forces.append(forces)
            del calc
            gc.collect()
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()

        energies = np.asarray(member_energies)
        for i, atoms in enumerate(atoms_list):
            stacked = np.stack([f[i] for f in member_forces])
            mean_forces = stacked.mean(axis=0)
            for j in range(len(members)):
                atoms.info[f"{name}_energy_m{j}"] = float(energies[j, i])
            atoms.info[f"{name}_energy"] = float(energies[:, i].mean())
            atoms.info[f"{name}_energy_std"] = float(energies[:, i].std())
            atoms.info[f"{name}_forces_std"] = float(
                np.linalg.norm(stacked.std(axis=0), axis=1).mean()
            )
            if ref_key in atoms.arrays:
                ref = atoms.arrays[ref_key]
                for j in range(len(members)):
                    atoms.info[f"{name}_forces_sse_m{j}"] = float(
                        ((stacked[j] - ref) ** 2).sum()
                    )
                atoms.info[f"{name}_forces_sse"] = float(
                    ((mean_forces - ref) ** 2).sum()
                )
            if write_forces:
                atoms.new_array(f"{name}_forces", mean_forces)
    return atoms_list


# --- host-side spec --------------------------------------------------------
class MaceEvalProcessor(Processor):
    """Score structures with named committees of torch MACE models.

    A pure host-side spec: it stages every member file once (shared across
    shards) and contributes the remote args; the compute is :func:`run`.
    ``registration_module`` defaults to this module, so the stock driver
    imports it and ``register`` runs -- ``torchsim_AL`` must be installed in
    the cluster env.

    Args:
        committees: ``{name: [model_path, ...]}``. ``name`` prefixes every
            output key; members are staged by basename, so basenames must be
            unique across all committees.
        ref_forces_key: ``arrays`` key of the reference forces to compute
            squared errors against (skipped for frames without it).
        device: Torch device on the cluster node.
        default_dtype: Precision the models are evaluated in.
        write_forces: Also write each committee's mean forces to ``arrays``.
    """

    name = "mace_eval"

    def __init__(
        self,
        committees: dict,
        *,
        ref_forces_key: str = "REF_forces",
        device: str = "cuda",
        default_dtype: str = "float64",
        write_forces: bool = False,
    ):
        if not committees:
            raise ValueError("MaceEvalProcessor needs at least one committee.")
        self.committees = {
            str(name): [Path(m) for m in members]
            for name, members in committees.items()
        }
        basenames = [m.name for ms in self.committees.values() for m in ms]
        if len(set(basenames)) != len(basenames):
            raise ValueError(
                "MaceEvalProcessor: model basenames must be unique, since "
                "they are staged side by side on the cluster."
            )
        self.ref_forces_key = ref_forces_key
        self.device = device
        self.default_dtype = default_dtype
        self.write_forces = bool(write_forces)

    def shared_files(self) -> list:
        return [str(m) for ms in self.committees.values() for m in ms]

    def build_args(self, savepath: str) -> dict:
        remotes = {
            name: [os.path.join(str(savepath), m.name) for m in members]
            for name, members in self.committees.items()
        }
        return MaceEvalArgs(
            committees=remotes,
            ref_forces_key=self.ref_forces_key,
            device=self.device,
            default_dtype=self.default_dtype,
            write_forces=self.write_forces,
        ).to_dict()


register("mace_eval", run, schema=MaceEvalArgs)
