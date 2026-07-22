"""Shared-body MACE ensemble model for torch_sim.

Wraps N finetuned MACE models into a single ModelInterface that runs the
expensive message-passing body once and applies each member's readout heads
separately.  Returns ensemble-mean energy/forces/stress and caches per-member
uncertainty (std) for downstream use.
"""

from __future__ import annotations

import copy
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
import torch_sim as ts
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl
from torch_sim.typing import StateLike

try:
    from mace.modules.models import MACE, ScaleShiftMACE
    from mace.modules.utils import (
        get_edge_vectors_and_lengths,
        get_outputs,
        get_symmetric_displacement,
    )
    from mace.tools import atomic_numbers_to_indices, utils
    from mace.tools.scatter import scatter_sum
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(
        f"MACE import failed: {traceback.format_exc()}", stacklevel=2
    )


def to_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generate one-hot encoding from indices.

    Copied from torch_sim MaceModel (pending upstream MACE merge).
    """
    shape = (*indices.shape[:-1], num_classes)
    oh = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)
    oh.scatter_(dim=-1, index=indices, value=1)
    return oh.view(*shape)


class MaceEnsembleModel(ModelInterface):
    """Shared-body MACE ensemble for computing mean predictions and uncertainty.

    Given N finetuned MACE models that share the same frozen body (embedding,
    interactions, product basis blocks) but have different readout heads,
    this wrapper:

    1. Runs the shared body **once** per forward call.
    2. Applies each member's readout heads + scale_shift + E0 separately.
    3. Returns ensemble-mean energy/forces/stress.
    4. Caches ``energy_uncertainty`` and ``forces_uncertainty`` (std across
       members) as attributes for downstream consumers.

    Attributes:
        n_members: Number of ensemble members.
        energy_uncertainty: Std of per-member energies, shape ``[n_systems]``.
            Set after each ``forward()`` call.
        forces_uncertainty: Std of per-member forces, shape ``[n_atoms, 3]``.
            Set after each ``forward()`` call.
    """

    def __init__(
        self,
        models: list[str | Path | torch.nn.Module],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Callable = torchsim_nl,
        compute_forces: bool = True,
        compute_stress: bool = True,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError(
                f"MaceEnsembleModel requires at least 2 models, got {len(models)}"
            )

        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.neighbor_list_fn = neighbor_list_fn
        self._memory_scales_with = "n_atoms_x_density"

        # -- Load all models ------------------------------------------------
        loaded: list[torch.nn.Module] = []
        for m in models:
            if isinstance(m, str | Path):
                mod = torch.load(
                    m, map_location=self.device, weights_only=False
                )
            elif isinstance(m, torch.nn.Module):
                mod = m
            else:
                raise TypeError(f"Expected path or nn.Module, got {type(m)}")
            mod = mod.to(device=self._device, dtype=self._dtype).eval()
            loaded.append(mod)

        # -- Validate compatibility -----------------------------------------
        self.is_scale_shift = isinstance(loaded[0], ScaleShiftMACE)
        ref = loaded[0]
        for i, mod in enumerate(loaded[1:], 1):
            if type(mod) is not type(ref):
                raise ValueError(
                    f"Model 0 is {type(ref).__name__} but model {i} is "
                    f"{type(mod).__name__}"
                )
            if float(mod.r_max) != float(ref.r_max):
                raise ValueError(
                    f"r_max mismatch: model 0 has {float(ref.r_max)}, "
                    f"model {i} has {float(mod.r_max)}"
                )
            if len(mod.interactions) != len(ref.interactions):
                raise ValueError(
                    f"Number of interactions mismatch: model 0 has "
                    f"{len(ref.interactions)}, model {i} has "
                    f"{len(mod.interactions)}"
                )

        # -- Store shared body from first model ----------------------------
        self.body = ref
        self.n_members = len(loaded)

        # -- Extract per-member head components ----------------------------
        self.member_readouts = torch.nn.ModuleList()
        self.member_atomic_energies = torch.nn.ModuleList()
        if self.is_scale_shift:
            self.member_scale_shifts = torch.nn.ModuleList()

        for mod in loaded:
            self.member_readouts.append(copy.deepcopy(mod.readouts))
            self.member_atomic_energies.append(
                copy.deepcopy(mod.atomic_energies_fn)
            )
            if self.is_scale_shift:
                self.member_scale_shifts.append(copy.deepcopy(mod.scale_shift))

        # -- Model metadata ------------------------------------------------
        self.r_max = ref.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in ref.atomic_numbers]
        )

        # -- Uncertainty caches --------------------------------------------
        self.energy_uncertainty: torch.Tensor | None = None
        self.forces_uncertainty: torch.Tensor | None = None
        self.stress_uncertainty: torch.Tensor | None = None
        self._uncertainty_report_counter = 0

        # -- Optionally lock atomic numbers at init ------------------------
        self.atomic_numbers_in_init = atomic_numbers is not None
        if atomic_numbers is not None:
            if system_idx is None:
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )
            self.setup_from_system_idx(atomic_numbers, system_idx)

    # -- ModelInterface required properties --------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    @property
    def memory_scales_with(self) -> str:
        return self._memory_scales_with

    # -- Setup (mirrors MaceModel) ----------------------------------------

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Configure internal batch bookkeeping from atomic numbers and
        system indices.  Mirrors ``MaceModel.setup_from_system_idx``."""
        self.atomic_numbers = atomic_numbers
        self.system_idx = system_idx

        self.n_systems = system_idx.max().item() + 1
        self.n_atoms_per_system: list[int] = []
        ptr = [0]
        for sys_idx in range(self.n_systems):
            n = int((system_idx == sys_idx).sum().item())
            self.n_atoms_per_system.append(n)
            ptr.append(ptr[-1] + n)
        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.total_atoms = int(atomic_numbers.shape[0])

        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(
                    atomic_numbers.cpu().numpy(), z_table=self.z_table
                ),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
            dtype=self.dtype,
        )

    # -- Forward -----------------------------------------------------------

    def forward(
        self, state: ts.SimState | StateLike
    ) -> dict[str, torch.Tensor]:
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(
                **state, masses=torch.ones_like(state["positions"])
            )
        )

        # --- atomic number validation (same logic as MaceModel) -----------
        if (
            sim_state.atomic_numbers is None
            and not self.atomic_numbers_in_init
        ):
            raise ValueError(
                "Atomic numbers must be provided in either the constructor "
                "or forward."
            )
        if (
            sim_state.atomic_numbers is not None
            and self.atomic_numbers_in_init
        ):
            raise ValueError(
                "Atomic numbers cannot be provided in both the constructor "
                "and forward."
            )
        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "System indices must be provided if not set during "
                    "initialization"
                )
            sim_state.system_idx = self.system_idx

        if (
            sim_state.atomic_numbers is not None
            and not self.atomic_numbers_in_init
            and not torch.equal(
                sim_state.atomic_numbers,
                getattr(
                    self, "atomic_numbers", torch.zeros(0, device=self.device)
                ),
            )
        ):
            self.setup_from_system_idx(
                sim_state.atomic_numbers, sim_state.system_idx
            )

        # --- Neighbor list (done once) ------------------------------------
        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            sim_state.positions,
            sim_state.row_vector_cell,
            sim_state.pbc,
            self.r_max,
            sim_state.system_idx,
        )
        shifts = ts.transforms.compute_cell_shifts(
            sim_state.row_vector_cell, unit_shifts, mapping_system
        )

        # --- Build data_dict (mirrors MaceModel) -------------------------
        positions = sim_state.positions
        cell = sim_state.row_vector_cell
        batch = sim_state.system_idx
        node_attrs = self.node_attrs

        positions.requires_grad_(True)
        node_attrs.requires_grad_(True)

        num_graphs = self.ptr.numel() - 1
        num_atoms_arange = torch.arange(positions.shape[0], device=self.device)
        # Each member uses head index 0 (single-head finetuned models)
        node_heads = torch.zeros_like(batch)

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        if self.compute_stress:
            positions, shifts, displacement = get_symmetric_displacement(
                positions=positions,
                unit_shifts=unit_shifts,
                cell=cell,
                edge_index=edge_index,
                num_graphs=num_graphs,
                batch=batch,
            )

        # =================================================================
        # SHARED BODY (run once)
        # =================================================================
        body = self.body

        # Node embedding
        node_feats = body.node_embedding(node_attrs)

        # Edge features
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=positions,
            edge_index=edge_index,
            shifts=shifts,
        )
        edge_attrs = body.spherical_harmonics(vectors)
        radial_out = body.radial_embedding(
            lengths, node_attrs, edge_index, body.atomic_numbers
        )
        # mace-freeze returns (edge_feats, cutoff); upstream returns tensor
        if isinstance(radial_out, tuple):
            edge_feats = radial_out[0]
        else:
            edge_feats = radial_out

        # Pair repulsion (shared)
        if hasattr(body, "pair_repulsion") and body.pair_repulsion:
            pair_node_energy = body.pair_repulsion_fn(
                lengths, node_attrs, edge_index, body.atomic_numbers
            )
        else:
            pair_node_energy = torch.zeros(
                positions.shape[0], device=self.device, dtype=self.dtype
            )

        # Interaction + product loop (shared)
        node_feats_per_layer: list[torch.Tensor] = []
        for interaction, product in zip(body.interactions, body.products):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=node_attrs
            )
            node_feats_per_layer.append(node_feats)

        # =================================================================
        # PER-MEMBER READOUT HEADS
        # =================================================================
        member_inter_es: list[torch.Tensor] = []
        member_total_es: list[torch.Tensor] = []

        for m in range(self.n_members):
            # E0 baseline per member
            node_e0_m = self.member_atomic_energies[m](node_attrs)[
                num_atoms_arange, node_heads
            ]
            e0_m = scatter_sum(
                src=node_e0_m, index=batch, dim=0, dim_size=num_graphs
            )

            # Readout heads per layer
            node_es_list = [pair_node_energy]
            for layer_idx, readout in enumerate(self.member_readouts[m]):
                node_es_list.append(
                    readout(node_feats_per_layer[layer_idx], node_heads)[
                        num_atoms_arange, node_heads
                    ]
                )

            # Sum interaction contributions
            node_inter_es_m = torch.sum(
                torch.stack(node_es_list, dim=0), dim=0
            )

            # Scale + shift (ScaleShiftMACE only)
            if self.is_scale_shift:
                node_inter_es_m = self.member_scale_shifts[m](
                    node_inter_es_m, node_heads
                )

            inter_e_m = scatter_sum(
                src=node_inter_es_m, index=batch, dim=-1, dim_size=num_graphs
            )
            total_e_m = e0_m + inter_e_m

            member_inter_es.append(inter_e_m)
            member_total_es.append(total_e_m)

        # =================================================================
        # ENSEMBLE MEAN + UNCERTAINTY
        # =================================================================
        inter_e_stack = torch.stack(member_inter_es, dim=0)  # [N, n_graphs]
        total_e_stack = torch.stack(member_total_es, dim=0)  # [N, n_graphs]

        mean_inter_e = inter_e_stack.mean(dim=0)  # [n_graphs]
        mean_total_e = total_e_stack.mean(dim=0)  # [n_graphs]

        # Energy uncertainty
        self.energy_uncertainty = total_e_stack.detach().std(dim=0)

        # --- Per-member forces (N backward passes) for uncertainty --------
        if self.compute_forces:
            member_forces: list[torch.Tensor] = []
            member_stresses: list[torch.Tensor] = []

            for m in range(self.n_members):
                # forces, virials, stress, hessian, edge_forces
                forces_m, _, stress_m, _, _ = get_outputs(
                    energy=member_inter_es[m],
                    positions=positions,
                    displacement=displacement,
                    cell=cell,
                    training=True,  # retain_graph for subsequent passes
                    compute_force=True,
                    compute_virials=self.compute_stress,
                    compute_stress=self.compute_stress,
                    compute_hessian=False,
                )
                member_forces.append(forces_m.detach())
                if self.compute_stress and stress_m is not None:
                    member_stresses.append(stress_m.detach())

            forces_stack = torch.stack(member_forces, dim=0)
            self.forces_uncertainty = forces_stack.std(dim=0)

            if self.compute_stress and member_stresses:
                stress_stack = torch.stack(member_stresses, dim=0)
                self.stress_uncertainty = stress_stack.std(dim=0)

        # --- Mean forces/stress (final backward, releases graph) ----------
        mean_forces, _, mean_stress, _, _ = get_outputs(
            energy=mean_inter_e,
            positions=positions,
            displacement=displacement,
            cell=cell,
            training=False,
            compute_force=self.compute_forces,
            compute_virials=self.compute_stress,
            compute_stress=self.compute_stress,
            compute_hessian=False,
        )

        # --- Reset counter for prop_calculators ---------------------------
        self._uncertainty_report_counter = 0

        # --- Build results ------------------------------------------------
        results: dict[str, torch.Tensor] = {
            "energy": mean_total_e.detach(),
        }
        if self.compute_forces and mean_forces is not None:
            results["forces"] = mean_forces.detach()
        if self.compute_stress and mean_stress is not None:
            results["stress"] = mean_stress.detach()

        if self.energy_uncertainty is not None:
            results["energy_uncertainty"] = self.energy_uncertainty
        if self.forces_uncertainty is not None:
            results["forces_uncertainty"] = self.forces_uncertainty
        if self.stress_uncertainty is not None:
            results["stress_uncertainty"] = self.stress_uncertainty

        return results

def _make_model(mace, device, dtype, compute_forces, compute_stress):
    """Create the appropriate model wrapper.

    If ``mace`` is a list (of paths or modules), returns a
    ``MaceEnsembleModel`` (shared-body ensemble). Otherwise wraps a single
    raw MACE module in ``MaceModel``.
    """
    if isinstance(mace, list):

        return MaceEnsembleModel(
            models=mace,
            device=device,
            dtype=dtype,
            compute_forces=compute_forces,
            compute_stress=compute_stress,
        )
    from torch_sim.models.mace import MaceModel
    
    return MaceModel(
        model=mace,
        device=device,
        dtype=dtype,
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )