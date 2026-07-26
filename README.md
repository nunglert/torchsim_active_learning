# torchsim_AL

A replica-exchange (parallel tempering) sampler built on [torch_sim](https://github.com/torchsim/torch-sim), designed to plug into `remote_handler`'s active-learning loop as a `Sampler`.

`TorchSimSampler` runs parallel-tempering MD across a ladder of temperatures with a MACE (or MACE-ensemble) model, submits it via `remote_handler`, collects the resulting trajectories, and hands back one `.extxyz` file of candidate configurations per AL iteration.

## Installation

```bash
pip install -e .
```

## Units — read this before setting `init_kwargs`/`step_kwargs`

The sampler automatically converts exactly three physical quantities from human-friendly units into torch_sim's internal "metal" unit system (Å, eV, amu):

| Sampler argument | You supply (physical units) | Converted internally via |
|---|---|---|
| `temperatures` | Kelvin | `kT = T_K * MetalUnits.temperature` |
| `dt` | picoseconds | `dt_native = dt_ps * MetalUnits.time` |
| `external_pressure` | bar | `P_native = P_bar * MetalUnits.pressure` |

**Everything else you put in `init_kwargs`/`step_kwargs` (`tau`, `gamma`, `alpha`, `cell_alpha`, `b_tau`, `t_tau`, `tau_p`, `isothermal_compressibility`) is passed straight through to torch_sim, unconverted — i.e. in torch_sim's native unit system, not ps/ps⁻¹/bar⁻¹.** If you want to reason about these in physical units, convert them yourself first:

```python
from torch_sim.units import MetalUnits

# "time" quantities (tau, b_tau, t_tau, tau_p): physical ps -> native
native_tau = tau_ps * MetalUnits.time

# "inverse time" quantities (gamma, alpha, cell_alpha): physical ps^-1 -> native
native_gamma = gamma_per_ps / MetalUnits.time

# "inverse pressure" quantities (isothermal_compressibility): physical bar^-1 -> native
native_beta_T = beta_T_per_bar / MetalUnits.pressure
```

If you omit any of these, torch_sim falls back to its own built-in defaults (usually expressed as a multiple of `dt`, e.g. "100·dt") — see the per-integrator tables below.

Do **not** put `external_pressure`, `dt`, `kT`, `state`, or `model` in `init_kwargs`/`step_kwargs` — they're supplied automatically by the sampler and will be rejected with an explicit error if you try (this also protects against the value you set being silently overwritten).

## Sampler parameters

| Parameter | Type | Units | Default | Notes |
|---|---|---|---|---|
| `remote_fs` | `FileSystemHandlerBase` | — | required | remote_handler machine handle |
| `cluster_config` | `ClusterConfig` | — | required | remote_handler partition/env/template config |
| `sampling_script` | path | — | required | path to `torch_sim_sampling.py` (or a compatible script) |
| `founder_path` | path | — | required | ASE-readable starting structure, local file |
| `temperatures` | `list[float]` | K | required | **must be strictly increasing**, ≥ 2 entries — replica exchange only swaps neighbor-indexed replicas (i, i+1), so an unsorted or duplicate ladder silently breaks the pairing |
| `integrator_name` | `str` | — | required | one of the 9 names in the integrator table below |
| `init_kwargs` | `dict` | integrator-specific, see below | required (can be `{}`) | validated against the real `*_init` signature |
| `step_kwargs` | `dict` | integrator-specific, see below | required (can be `{}`) | validated against the real `*_step` signature |
| `target_temperature` | `float \| None` | K | `None` | informational only — not consumed by the sampler itself |
| `model_paths` | `list[str] \| None` | — | `["foundation"]` | overridden every iteration by whatever `update()` receives as `sampler_input` from the AL loop |
| `dt` | `float` | ps | `0.001` | integration timestep |
| `external_pressure` | `float \| None` | bar | `None` | NPT only; must be `None` or `0` for NVT integrators. If left `None` for NPT, defaults to 0 bar (logged as a warning) |
| `n_steps_npt` | `int` | steps | `0` | **despite the name, this applies to both NVT and NPT.** It does *not* run a separate equilibration phase — it only controls how many leading MD steps are excluded when `get_sampled_configs()` selects frames for the AL loop. Must be `< n_steps_pt` |
| `n_steps_pt` | `int` | steps | `100` | total production PT steps |
| `seed` | `int` | — | `123` | RNG seed |
| `device` | `str` | — | `"cuda"` | `"cpu"`, `"cuda"`, or `"cuda:N"` |
| `dtype` | `str` | — | `"float32"` | `"float32"` or `"float64"` |
| `remote_savepath` | path \| `None` | — | `None` | remote-side calculation directory |
| `loop_time` | `float` | seconds | `30` | polling interval for job completion |
| `stub` | `bool` | — | `False` | skip cluster submission, generate mock output instead. Input validation still runs in stub mode — only the remote upload/submit is skipped |
| `md_steps_per_exchange` | `int` | steps | `100` | MD steps between replica-exchange attempts |
| `write_frequency` | `int` | steps | `10` | how often trajectory frames + per-step properties (energy, uncertainty) are written. Must be `<= n_steps_pt` |
| `swap_temperature` | `bool` | — | `True` | whether exchange attempts swap temperature labels |
| `rescale_momenta_on_swap` | `bool` | — | `True` | rescale momenta to the new temperature label on an accepted swap |
| `rescale_barostat_on_swap` | `bool` | — | `True` | also rescale NPT barostat DOFs on an accepted swap (no-op for NVT) |

`model_paths` / `sampler_input` accepts either `["foundation"]` (MACE-MP foundation model) or a list of local paths to MACE model files. A single model path runs a standard simulation; **two or more** activates `MaceEnsembleModel` (shared message-passing body, per-member readout heads, ensemble-mean energy/forces + `energy_uncertainty`/`forces_uncertainty` attached to every output structure).

### Kwarg reference (name → meaning, native units, default)

All defaults below are torch_sim's own fallback when the kwarg is omitted, expressed relative to `dt` (in native units, i.e. **after** the ps→native conversion already applied to `dt`).

| Kwarg | Used by | Meaning | Native units | Default if omitted |
|---|---|---|---|---|
| `tau` (step) | `nvt_vrescale` | V-Rescale thermostat relaxation time | time | `100 * dt` |
| `tau` (step) | `npt_crescale_*` | V-Rescale thermostat relaxation time (temperature control) | time | `100 * dt` |
| `gamma` | `nvt_langevin` | Langevin friction coefficient | 1/time | `1 / (100 * dt)` |
| `tau` (init) | `nvt_nose_hoover` | Nosé–Hoover chain relaxation time | time | `100 * dt` |
| `chain_length` | `nvt_nose_hoover`, `npt_nose_hoover_isotropic` | number of thermostats in the Nosé–Hoover chain | — | `3` |
| `chain_steps` | `nvt_nose_hoover` | chain integration substeps | — | `3` |
| `chain_steps` | `npt_nose_hoover_isotropic` | chain integration substeps | — | `2` |
| `sy_steps` | `nvt_nose_hoover`, `npt_nose_hoover_isotropic` | Suzuki–Yoshida integration order (must be 1, 3, 5, or 7) | — | `3` |
| `alpha` | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | particle friction | 1/time | `1 / (5 * dt)` |
| `cell_alpha` | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | cell friction | 1/time | `1 / (30 * dt)` |
| `b_tau` | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | barostat time constant | time | `300 * dt` |
| `t_tau` | `npt_nose_hoover_isotropic` | thermostat (temperature) relaxation time | time | `100 * dt` |
| `b_tau` | `npt_nose_hoover_isotropic` | barostat (pressure) relaxation time | time | `1000 * dt` |
| `tau_p` | `npt_crescale_*` | barostat relaxation time | time | `3 * dt` |
| `isothermal_compressibility` | `npt_crescale_*` | isothermal compressibility | 1/pressure | `1e-6 bar⁻¹` (converted internally) |