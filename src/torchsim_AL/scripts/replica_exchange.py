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
    swap_temperature: bool = True,
    swap_pressure: bool = False,
    rescale_momenta: bool = True,
    rescale_barostat: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    """
    General replica exchange move for NVT / NPT.

    Supports:
      - temperature swap (kT labels) for NVT and NPT
      - pressure swap (P labels) for NPT (optional)
      - both simultaneously

    Acceptance uses:
      S(x; beta, P) = beta * (U(x) + P*V(x))   (isotropic P)
    so log r = -[S(x_i; beta_j,P_j)+S(x_j; beta_i,P_i) - S(x_i; beta_i,P_i)-S(x_j; beta_j,P_j)].

    Returns updated (kT, external_pressure, accepted).
    """
    U = state.energy
    if U.ndim != 1:
        U = U.view(-1)

    n_sys = int(U.shape[0])

    # Handle pressure parameterization, only isotropic case considered
    P = external_pressure
    if not isinstance(P, torch.Tensor):
        P = torch.as_tensor(P, device=kT.device, dtype=kT.dtype)
    # Accept scalar or [n_systems]; (ignore tensor anisotropy here)
    if P.ndim == 0:
        P = P.expand(n_sys)
    elif P.ndim == 3:
        # if [n_systems,3,3], interpret as isotropic via trace/3
        P = torch.einsum("bii->b", P) / 3.0
    P_i = P[i]
    P_j = P[j]

    V = _get_volume_per_system(state) if P is not None else None

    beta = 1.0 / kT
    beta_i, beta_j = beta[i], beta[j]

    # current labels
    kTi, kTj = kT[i], kT[j]

    # proposed labels
    kT_i_new, kT_j_new = (kTj, kTi) if swap_temperature else (kTi, kTj)

    def S(U, V, beta, P):
        return beta * (U + P * V)

    # log r = -[S_i(new) + S_j(new) - S_i(old) - S_j(old)]
    if P is None:
        log_r = -(
            S(U[i], None, beta_j if swap_temperature else beta_i, None)
            + S(U[j], None, beta_i if swap_temperature else beta_j, None)
            - S(U[i], None, beta_i, None)
            - S(U[j], None, beta_j, None)
        )
    else:
        # old actions
        S_i_old = S(U[i], V[i], beta_i, P_i)
        S_j_old = S(U[j], V[j], beta_j, P_j)
        # new actions with swapped labels
        beta_i_new = 1.0 / kT_i_new
        beta_j_new = 1.0 / kT_j_new
        S_i_new = S(U[i], V[i], beta_i_new, P)
        S_j_new = S(U[j], V[j], beta_j_new, P)
        log_r = -(S_i_new + S_j_new - S_i_old - S_j_old)

    u = torch.rand((), generator=rng, device=kT.device, dtype=kT.dtype)
    accept = torch.log(u) < log_r
    if not bool(accept):
        return kT, external_pressure, False

    # apply swaps
    kT_new = kT.clone()
    if swap_temperature:
        kT_new[i], kT_new[j] = kTj, kTi

    # rescale kinetic DOFs when temperature label changes
    if rescale_momenta and swap_temperature:
        si = torch.sqrt(kT_new[i] / kTi)
        sj = torch.sqrt(kT_new[j] / kTj)
        _rescale_atom_momenta_in_place(state, i, si)
        _rescale_atom_momenta_in_place(state, j, sj)
        if rescale_barostat:
            _rescale_barostat_dofs_in_place(state, i, si)
            _rescale_barostat_dofs_in_place(state, j, sj)
    return kT_new, external_pressure, True