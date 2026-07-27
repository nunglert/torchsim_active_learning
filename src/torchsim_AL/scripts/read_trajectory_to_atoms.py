import numpy as np
from ase import Atoms
from torch_sim.trajectory import TorchSimTrajectory
from torch_sim.io import state_to_atoms
import logging

logger = logging.getLogger(__name__)

def _read_trajectory_frames(traj, n_steps_npt, write_frequency) -> list[Atoms]:
    """Read all frames from a TorchSimTrajectory, attaching custom properties.

    ``state_to_atoms`` only returns basic structural data
    (positions, cell, numbers, pbc).  This helper additionally reads any
    custom arrays stored by ``TrajectoryReporter`` (e.g. from
    ``prop_calculators``) and attaches them to each ``Atoms`` object:

    * Scalar-per-frame arrays (like ``energy_uncertainty``) are placed in
      ``atoms.info[name]``.
    * Per-atom arrays whose first dimension matches the atom count (like
      ``forces_uncertainty`` with shape ``[n_atoms, 3]``) are placed in
      ``atoms.arrays[name]``.
    """

    n_frames = len(traj)
    burn_in = n_steps_npt // max(write_frequency, 1)
    
    frames = [state_to_atoms(traj.get_state(i))[0] for i in range(burn_in, n_frames)]

    if burn_in >= n_frames:
            logger.warning(
                "n_steps_npt=%d (%d frames at write_frequency=%d) covers the "
                "entire trajectory (%d frames) — no production frames to return.",
                n_steps_npt,
                burn_in,
                write_frequency,
                n_frames,
            )

    # State arrays written by write_state — skip these when attaching props
    _state_keys = {"positions", "cell", "masses", "atomic_numbers", "pbc"}
    prop_names = [k for k in traj.array_registry if k not in _state_keys]

    for name in prop_names:
        try:
            arr = traj.get_array(name=name, start=burn_in, stop=n_frames)
        except Exception:
            continue

        for i, at in enumerate(frames):
            if i >= arr.shape[0]:
                break
            val = arr[i]
            if val.ndim == 0 or (val.ndim == 1 and val.shape[0] == 1):
                # Scalar → atoms.info
                at.info[name] = val.item()
            elif val.ndim >= 1 and val.shape[0] == len(at):
                # Per-atom array → atoms.arrays
                at.arrays[name] = np.ascontiguousarray(val)
            else:
                at.info[name] = val

    for at in frames:
        at.wrap()

    return frames