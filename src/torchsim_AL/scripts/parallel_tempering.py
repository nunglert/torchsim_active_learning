import torch
import torch_sim as ts
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.units import MetalUnits
from torch_sim.state import SimState
from torch_sim import runners
from torch_sim.integrators import INTEGRATOR_REGISTRY
from torchsim_AL.scripts.replica_exchange import attempt_replica_exchange
import logging

logger = logging.getLogger(__name__)

"""Replica-exchange (parallel tempering) driver on top of torch_sim.

Runs n_systems replicas (one per temperature) as a single batched
simulation, alternating blocks of ordinary MD (md_steps_per_exchange steps)
with a round of neighbor-pair exchange attempts (attempt_replica_exchange).
No atoms are ever physically moved between replicas — only kT labels swap.
"""

def parallel_tempering(
    replica_atoms,
    model,
    integrator_name,
    temperatures: torch.Tensor,
    dt: float,
    init_kwargs,
    step_kwargs,
    dtype,
    device,
    n_steps_npt: int,
    n_steps_pt: int,
    md_steps_per_exchange: int = 100,
    seed: int = 42,
    external_pressure: torch.Tensor | float | None = 0,
    swap_temperature: bool = True,
    rescale_momenta_on_swap: bool = True,
    rescale_barostat_on_swap: bool = True,
    reporter_args=None,
    log_every_n_rounds: int = 10,
):
    
    """Run replica-exchange MD (equilibration + production) and return the
    final state and swap log.

    Args:
        replica_atoms: One ASE Atoms per replica (e.g. identical copies of a
            founder structure) — converted to a batched SimState internally
            via ts.initialize_state(replica_atoms, device, dtype).
        model: torch_sim ModelInterface (e.g. MaceModel, MaceEnsembleModel).
        integrator_name: A torch_sim.Integrator name, e.g. "nvt_langevin".
        temperatures: Per-replica temperatures in K (human-readable; this
            function converts to kT in torch_sim's native units internally).
            Must be increasing — exchange only proposes swaps
            between neighbor-indexed replicas i, i+1.
        dt: Timestep in ps (human-readable; converted to native units here).
        init_kwargs, step_kwargs: Forwarded to the integrator's init/step
            functions; see README for what's valid per integrator.
        dtype, device: Target torch dtype/device for the whole run — the
            state, temperatures, and dt are all built onto these directly,
            not inferred from the model or from `temperatures`'s own type.
        n_steps_npt: Equilibration steps run before the first exchange
            attempt (no swaps proposed during this phase). Despite the
            name, applies regardless of ensemble (NVT or NPT).
        n_steps_pt: Production steps run after equilibration, with the
            normal alternating MD-block/exchange-attempt cycle.
            Total simulated length is n_steps_npt + n_steps_pt.
        md_steps_per_exchange: MD steps per block between exchange attempts
            during the production phase.
        seed: Seeds this function's own exchange accept/reject draws (via a
            dedicated torch.Generator) AND the state's own RNG (via
            initial_state.rng = seed) — so with the same seed, both the
            exchange decisions and the underlying MD trajectory (initial
            velocities, thermostat noise) are fully reproducible.
        external_pressure: None/0 for NVT. For NPT, target pressure in bar
            (human-readable) — converted to native units internally as P;
            never mutated, only read.
        swap_temperature: If False, no exchange attempts are made at all —
            replicas run as independent MD at fixed temperatures for the
            whole run (equilibration + production).
        rescale_momenta_on_swap, rescale_barostat_on_swap: See
            attempt_replica_exchange.
        reporter_args: kwargs for torch_sim.TrajectoryReporter, or None to
            skip trajectory writing. Written continuously across both the
            equilibration and production phases — the equilibration/
            production split only affects when swaps start, not what gets
            recorded; separating the two happens later when reading the
            trajectory back (see read_trajectory_to_atoms.py).

    Returns:
        (state, temperatures, external_pressure, swap_log) — swap_log has
        "data" (per-attempt records) and "info" (summary stats including
        accept_rate). Note temperatures/external_pressure here are your
        original human-readable inputs passed straight through, not the
        internally-converted kT/P values.
    """

    #n_sys specified for replica exchange
    n_sys = len(replica_atoms)

    #Set up seed for replica exchange
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    #Set up the SimState with seed
    initial_state: SimState = ts.initialize_state(replica_atoms, device, dtype)
    initial_state.rng = seed

    #Determine correct size of first simulation block
    start_swap = n_steps_npt
    n_total_steps = n_steps_pt + n_steps_npt
    first_block = max(min(start_swap, n_total_steps), min(md_steps_per_exchange, n_total_steps))

    # Temperatures: human-readable Kelvin in, native energy units (kT) out
    temperatures_tensor = torch.tensor(
        temperatures, dtype=dtype, device=device
    )
    kT = temperatures_tensor * MetalUnits.temperature
    kTs = runners._normalize_temperature_tensor(temperature=kT, n_steps=first_block, initial_state=initial_state)

    # Pressure: external_pressure stays human-readable (bar) as passed in;
    # P is the torch_sim-native converted value used everywhere below.
    if integrator_name.startswith("npt"):
        P = (external_pressure or 0) * MetalUnits.pressure
        step_kwargs["external_pressure"] = P
    else:
        external_pressure = None
        P = None

    #Set up dt (ps -> native time units)
    dt = torch.tensor(dt * MetalUnits.time, dtype=dtype, device=device)

    #Set up integrator
    integrator = getattr(ts.Integrator, integrator_name)
    init_func, step_func = INTEGRATOR_REGISTRY[integrator]

    #Inititalize MDState with initial momenta etc.
    state = init_func(
        state=initial_state, model=model, kT=kTs[0], dt=dt, **init_kwargs
        )

    #Set up Reporter
    if reporter_args is not None:
        mode = "w"
        ra = dict(reporter_args)
        tk = ra.pop("trajectory_kwargs", {})
        tk["mode"] = mode
        ra["trajectory_kwargs"] = tk
        trajectory_reporter = TrajectoryReporter(**ra)
    else:
        trajectory_reporter = None
    runners._write_initial_state(trajectory_reporter, state, model)

    #Run simulation in blocks and swap in between
    try:
        stats = {
            "attempted": 0,
            "accepted": 0,
            "accept_rate": 0.0,
            "attempted_pairs": [],  # (step, i, j, accepted)
        }

        steps_done = 0
        exchange_round = 0

        while steps_done < n_total_steps:
            if exchange_round == 0:
                block = first_block
            else:
                block = min(md_steps_per_exchange, n_total_steps - steps_done)

            initial_step = steps_done + 1
            for step in range(initial_step, initial_step + block):
                state = step_func(
                    state=state,
                    model=model,
                    dt=dt,
                    kT=kTs[step - initial_step],
                    **step_kwargs,
                )
                state.positions = state.positions.detach()
                if trajectory_reporter:
                    trajectory_reporter.report(state, step, model=model)
            steps_done += block
            if swap_temperature:
                # choose non-overlapping neighbor pairs
                if (exchange_round % 2) == 0:
                    pairs = [(i, i + 1) for i in range(0, n_sys - 1, 2)]
                else:
                    pairs = [(i, i + 1) for i in range(1, n_sys - 1, 2)]

                # attempt exchanges
                for i, j in pairs:
                    stats["attempted"] += 1
                    kT, P, acc = attempt_replica_exchange(
                        state,
                        i=i,
                        j=j,
                        rng=rng,
                        kT=kT,
                        external_pressure=P,
                        rescale_momenta=rescale_momenta_on_swap,
                        rescale_barostat=rescale_barostat_on_swap,
                    )
                    stats["accepted"] += int(acc)
                    stats["attempted_pairs"].append(
                        (steps_done, i, j, bool(acc))
                    )

            # update temperature labels for next MD block
            kTs = runners._normalize_temperature_tensor(temperature=kT, n_steps=md_steps_per_exchange, initial_state=initial_state)
            exchange_round += 1
            if exchange_round == 1 or exchange_round % log_every_n_rounds == 0:
                logger.info(
                    "PT progress: step %d/%d (%.0f%%), round %d, "
                    "accept_rate so far=%.3f (%d/%d attempted)",
                    steps_done, n_total_steps, 100 * steps_done / n_total_steps,
                    exchange_round,
                    stats["accepted"] / max(1, stats["attempted"]),
                    stats["accepted"], stats["attempted"],
                )

        stats["accept_rate"] = stats["accepted"] / max(1, stats["attempted"])

        swap_data = [
            {"step": int(step), "i": int(i), "j": int(j), "accepted": bool(acc)}
            for (step, i, j, acc) in stats["attempted_pairs"]
        ]
        swap_log = {
            "data": swap_data,
            "info": {
                "swap_temperature": bool(swap_temperature),
                "accept_rate": float(stats["accept_rate"]),
                "attempted": int(stats["attempted"]),
                "accepted": int(stats["accepted"]),
                "temperatures": temperatures,
                **({"external_pressure": P.detach().cpu().tolist() if isinstance(P, torch.Tensor) else P} if P is not None else {}),
                },
            }

    finally:
        # Always flush/close, even if an exception (e.g. NaN forces) cuts
        # the run short — otherwise the partial trajectory can be left with
        # unflushed HDF5 buffers, making a failed job even harder to debug.
        if trajectory_reporter:
            trajectory_reporter.finish()

    return state, temperatures, P, swap_log