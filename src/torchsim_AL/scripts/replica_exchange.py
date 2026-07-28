import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_volume_per_system(state) -> torch.Tensor:
    """
    Returns volume per system, shape [n_systems].
    Tries state.cell first; falls back to state.current_cell if present.
    """
    return torch.linalg.det(state.cell)


def _rescale_atom_momenta_in_place(state, sys: int, scale: torch.Tensor):
    mask = state.system_idx == sys
    p = state.momenta.clone()
    p[mask] *= scale
    if hasattr(state, "set_constrained_momenta"):
        state.set_constrained_momenta(p)
    else:
        state.momenta = p


def _rescale_barostat_dofs_in_place(state, sys: int, scale: torch.Tensor):
    """
    Best-effort rescaling of barostat DOFs for Nose–Hoover style NPT states.
    Safe no-op for states that don't have these attributes.
    """
    # Nose–Hoover NPT in your code uses: cell_momentum [n_systems,1] and chain momenta
    if hasattr(state, "cell_momentum") and state.cell_momentum is not None:
        cm = state.cell_momentum.clone()
        # cm shape could be [n_systems] or [n_systems,1]
        cm[sys] *= scale
        state.cell_momentum = cm

    # If you ever store a full cell-velocity tensor (Langevin NPT), you can rescale it too
    if hasattr(state, "cell_velocities") and state.cell_velocities is not None:
        cv = state.cell_velocities.clone()
        cv[sys] *= scale
        state.cell_velocities = cv


def attempt_replica_exchange(
    state,
    *,
    i: int,
    j: int,
    rng: torch.Generator,
    kT: torch.Tensor,  # [n_systems]
    external_pressure: torch.Tensor
    | None = None,  # scalar or [n_systems] or [n_systems,3,3]
    rescale_momenta: bool = True,
    rescale_barostat: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """
    Replica exchange move for NVT / NPT.

    Acceptance uses:
      S(x; beta, P) = beta * (U(x) + P*V(x))   (isotropic P)

    Returns updated (kT, external_pressure, accepted).
    """
    U = state.energy
    if U.ndim != 1:
        U = U.view(-1)

    n_sys = int(U.shape[0])

    beta = 1.0 / kT
    beta_i, beta_j = beta[i], beta[j]

    kT_i, kT_j = kT[i], kT[j]
    beta_i_new = beta_j
    beta_j_new = beta_i

    def S(U, beta, P=None, V=None):
        s = beta * U
        if P is not None:
            s = s + beta * P * V
        return s

    if external_pressure is None:
        # Pure NVT acceptance criterion, no P*V term. torch.as_tensor(None)
        # raises, so this must be handled before any tensor conversion below.
        S_i_old = S(U[i], beta_i)
        S_j_old = S(U[j], beta_j)
        S_i_new = S(U[i], beta_i_new)
        S_j_new = S(U[j], beta_j_new)
    else:
        # Handle pressure parameterization, only isotropic case considered.
        P = external_pressure
        if not isinstance(P, torch.Tensor):
            P = torch.as_tensor(P, device=kT.device, dtype=kT.dtype)
        # Accept scalar or [n_systems]; (ignore tensor anisotropy here)
        if P.ndim == 0:
            P = P.expand(n_sys)
        elif P.ndim == 3:
            # if [n_systems,3,3], interpret as isotropic via trace/3
            P = torch.einsum("bii->b", P) / 3.0
        P_i, P_j = P[i], P[j]
        V = _get_volume_per_system(state)
        V_i, V_j = V[i], V[j]

        S_i_old = S(U[i], beta_i, P_i, V_i)
        S_j_old = S(U[j], beta_j, P_j, V_j)
        S_i_new = S(U[i], beta_i_new, P_i, V_i)
        S_j_new = S(U[j], beta_j_new, P_j, V_j)

    log_r = -(S_i_new + S_j_new - S_i_old - S_j_old)

    u = torch.rand((), generator=rng, device=kT.device, dtype=kT.dtype)
    accept = torch.log(u) < log_r
    if not bool(accept):
        return kT, external_pressure, False

    # apply swaps
    kT_new = kT.clone()
    kT_new[i], kT_new[j] = kT_j, kT_i

    # rescale kinetic DOFs when temperature label changes
    if rescale_momenta:
        si = torch.sqrt(kT_new[i] / kT_i)
        sj = torch.sqrt(kT_new[j] / kT_j)
        _rescale_atom_momenta_in_place(state, i, si)
        _rescale_atom_momenta_in_place(state, j, sj)
        if rescale_barostat:
            _rescale_barostat_dofs_in_place(state, i, si)
            _rescale_barostat_dofs_in_place(state, j, sj)
    return kT_new, external_pressure, True