import torch
import torch_sim as ts

def _make_uncertainty_prop(attr, advance=False):
    """Create a prop_calculator function that reads cached uncertainty from model.

    The ``MaceEnsembleModel`` caches batch-level uncertainty tensors
    (``energy_uncertainty`` shape ``[n_systems]``, ``forces_uncertainty``
    shape ``[n_atoms, 3]``) after every ``forward()`` call.  Since the
    ``TrajectoryReporter`` calls prop fns per-substate (one system at a
    time, in order), we use ``_uncertainty_report_counter`` on the model to
    index into the batch tensors.

    For per-atom tensors (``forces_uncertainty``) the correct atom slice is
    determined from ``model.ptr``.

    Args:
        attr: Name of the cached uncertainty attribute on the model.
        advance: If True this prop fn increments the counter after reading.
            Exactly **one** prop fn per reporting frequency must set this
            to True, and it should be the last one inserted into the dict
            (Python 3.7+ guarantees insertion-order iteration).
    """

    def fn(state, model):
        unc = getattr(model, attr, None)
        if unc is None:
            return torch.tensor(0.0, device=model.device)
        idx = model._uncertainty_report_counter
        if unc.dim() == 1 and unc.shape[0] > 1:
            # Per-system scalar (e.g. energy_uncertainty [n_systems])
            val = unc[idx]
        elif unc.dim() == 2:
            # Per-atom tensor (e.g. forces_uncertainty [n_atoms, 3])
            start = model.ptr[idx]
            end = model.ptr[idx + 1]
            val = unc[start:end]
        else:
            val = unc
        if advance:
            model._uncertainty_report_counter += 1
        return val

    return fn


def _build_prop_calculators(model, frequency=1):
    """Build prop_calculators dict, adding uncertainty entries for ensembles.

    Returns a dict suitable for passing to ``TrajectoryReporter``.
    When *model* is a ``MaceEnsembleModel``, energy and force uncertainty
    entries are appended.  These read from the model's cache (no
    recomputation) using ``_make_uncertainty_prop``.
    """
    from torchsim_AL.scripts.make_mace import (
        MaceEnsembleModel,
    )

    props = {
        frequency: {
            "potential_energy": lambda state, model: state.energy,
            "kinetic_energy": lambda state, model: ts.calc_kinetic_energy(
                momenta=state.momenta,
                masses=state.masses,
            ),
        },
    }

    if isinstance(model, MaceEnsembleModel):
        props[frequency]["energy_uncertainty"] = _make_uncertainty_prop(
            "energy_uncertainty",
        )
        # Last entry → advance=True increments counter
        props[frequency]["forces_uncertainty"] = _make_uncertainty_prop(
            "forces_uncertainty",
            advance=True,
        )

    return props