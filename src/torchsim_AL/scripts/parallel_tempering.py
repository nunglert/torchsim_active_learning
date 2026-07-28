import torch
import torch_sim as ts
from torch_sim.trajectory import TrajectoryReporter
from torch_sim.units import MetalUnits
from torch_sim.state import SimState
from torch_sim import runners
from torch_sim.integrators import INTEGRATOR_REGISTRY
from torchsim_AL.scripts.replica_exchange import attempt_replica_exchange

def parallel_tempering(
    state,
    model,
    integrator_name,
    temperatures: torch.Tensor,
    dt: float,
    init_kwargs,
    step_kwargs,
    n_total_steps: int,
    md_steps_per_exchange: int = 100,
    seed: int = 42,
    external_pressure: torch.Tensor | float | None = 0,
    swap_temperature: bool = True,
    rescale_momenta_on_swap: bool = True,
    rescale_barostat_on_swap: bool = True,
    start_step: int = 0,
    reporter_args=None,
):
    """
    Parallel tempering / replica exchange for batched simulations.

    Notes
    -----
    - We do NOT shuffle atoms between replicas. We swap labels.
    """

    device = temperatures.device
    dtype = temperatures.dtype
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    n_sys = int(state.n_systems)
    assert (
        temperatures.shape[0] == n_sys
    ), f"Expected temperatures shape [{n_sys}], got {tuple(temperatures.shape)}"

    # Labels (kT in energy units)
    kT = temperatures * MetalUnits.temperature

    # Pressure labels (only for NPT); allow scalar or per-replica
    if integrator_name.startswith("npt"):
        external_pressure = (external_pressure or 0) * MetalUnits.pressure
        P = external_pressure
        step_kwargs["external_pressure"] = external_pressure
    else:
        P = None

    integrator = getattr(ts.Integrator, integrator_name)
    init_func, step_func = INTEGRATOR_REGISTRY[integrator]
    initial_state: SimState = ts.initialize_state(state, model.device, model.dtype)
    initial_state.rng = seed
    first_block = max(min(start_step, n_total_steps), min(md_steps_per_exchange, n_total_steps))
    kTs = runners._normalize_temperature_tensor(temperature=kT, n_steps=first_block, initial_state=initial_state)
    dt = torch.tensor(dt * MetalUnits.time, dtype=dtype, device=device)
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
                # --- choose non-overlapping neighbor pairs
                if (exchange_round % 2) == 0:
                    pairs = [(i, i + 1) for i in range(0, n_sys - 1, 2)]
                else:
                    pairs = [(i, i + 1) for i in range(1, n_sys - 1, 2)]

                # --- attempt exchanges
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

            # --- update temperature labels in Kelvin for next MD block
            kTs = runners._normalize_temperature_tensor(temperature=kT, n_steps=md_steps_per_exchange, initial_state=initial_state)
            exchange_round += 1

        stats["accept_rate"] = stats["accepted"] / max(1, stats["attempted"])

        # swap_log in your previous format
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
                "temperatures": temperatures.detach().cpu().tolist(),
                **({"external_pressure": P.detach().cpu().tolist()} if P is not None else {}),
                },
            }

    finally:
        # Always flush/close, even if an exception (e.g. NaN forces) cuts
        # the run short — otherwise the partial trajectory can be left with
        # unflushed HDF5 buffers, making a failed job even harder to debug.
        if trajectory_reporter:
            trajectory_reporter.finish()

    return state, temperatures, P, swap_log