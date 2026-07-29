# torchsim_AL

A replica-exchange (parallel tempering) sampler built on [torch_sim](https://github.com/torchsim/torch-sim), designed to plug into `remote_handler`'s active-learning loop as a `Sampler`.

`TorchSimSampler` runs parallel-tempering MD across a ladder of temperatures with a MACE (or MACE-ensemble) model, submits it via `remote_handler`, collects the resulting trajectories, and hands back one `.extxyz` file of candidate configurations per AL iteration.

## Installation

Installation requires private dependencies as specified in `pyproject.toml`.

```bash
pip install -e .
```

If you are planning on using selection based on QUESTS for your active learning loop use this.

```bash
pip install -e .[quests]
```
## Quickstart

This sets up an instance of the `TorchSimSampler`. Using `update()`, the sampler gets supplied with an output path and a MACE model for an active learning iteration. `get_sampled_configs()` uses the `remote_handler` to put all relevant files (model, parallel tempering script, input files, slurm script,...) onto a cluster, execute the remote task, collect the sampled configurations to the previously specified path and returns that path. A more elaborate example for an active learning loop using `remote_handler` is provided in `example/run_torchsim_al.py`.

```
from pathlib import Path
from torchsim_AL.sampler import TorchSimSampler

sampler = TorchSimSampler(
    remote_fs=None, cluster_config=None,           # not needed in stub mode
    sampling_script="unused.py",
    founder_path="structure.xyz",
    temperatures=[300, 400, 550],
    integrator_name="nvt_langevin",
    init_kwargs={}, step_kwargs={"gamma": 1.0},
    stub=True,
)
sampler.update(calcdir=Path("iter0"), sampler_input=["foundation"])
samples_path = sampler.get_sampled_configs()
```

## Units

The sampler automatically converts exactly three physical quantities from human-friendly units into torch_sim's internal "metal" unit system (Å, eV, amu):

| Sampler argument | You supply (physical units) | Converted internally via |
|---|---|---|
| `temperatures` | Kelvin | `kT = T_K * MetalUnits.temperature` |
| `dt` | picoseconds | `dt_native = dt_ps * MetalUnits.time` |
| `external_pressure` | bar | `P_native = P_bar * MetalUnits.pressure` |

**Everything else you put in `init_kwargs`/`step_kwargs` is passed straight through to torch_sim, unconverted — i.e. in torch_sim's native unit system, not ps/ps⁻¹/bar⁻¹.**

Do **not** put `external_pressure`, `dt`, `kT`, `state`, or `model` in `init_kwargs`/`step_kwargs` — they're supplied automatically by the sampler and will be rejected with an explicit error if you try (this also protects against the value you set being silently overwritten).

## Sampler parameters

Parameters get validated before any files get put on a cluster or any tasks get executed via `sampler_spec.validate_spec`, so that runs fail early. Logging returns useful warnings and errors if they arise.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `remote_fs` | `FileSystemHandlerBase` | required | remote_handler machine handle |
| `cluster_config` | `ClusterConfig` | required | remote_handler partition/env/template config |
| `sampling_script` | path | required | path to `torch_sim_sampling.py` (or a compatible script) |
| `founder_path` | path | required | ASE-readable starting structure, local file |
| `temperatures` | `list[float]` | required | K. **must be increasing**, ≥ 2 entries — replica exchange only swaps neighbor-indexed replicas (i, i+1) |
| `integrator_name` | `str` | required | one of the 9 names in the integrator table below |
| `init_kwargs` | `dict` | required (can be `{}`) | integrator-specific, see below; validated against the real `*_init` signature |
| `step_kwargs` | `dict` | required (can be `{}`) | integrator-specific, see below; validated against the real `*_step` signature |
| `model_paths` | `list[str] \| None` | `["foundation"]` | overridden every iteration by whatever `update()` receives as `sampler_input` from the AL loop |
| `dt` | `float` | `0.001` | ps. integration timestep |
| `external_pressure` | `float \| None` | `None` | bar. NPT only; must be `None` or `0` for NVT integrators. If left `None` for NPT, defaults to 0 bar (logged as a warning) |
| `n_steps_equil` | `int` | `0` | steps. **despite the name, this applies to both NVT and NPT.** It does *not* run a separate equilibration phase — it only controls how many leading MD steps are excluded when `get_sampled_configs()` selects frames for the AL loop and excluded from exchange attempts. Must be `< n_steps_prod` |
| `n_steps_prod` | `int` | `100` | steps. production PT steps. total steps = `n_steps_equil` + `n_steps_prod` |
| `seed` | `int` | `42` | RNG seed. Subsequent iterations increment up by 1 from this seed. |
| `device` | `str` | `"cuda"` | `"cpu"`, `"cuda"`, or `"cuda:N"` |
| `dtype` | `str` | `"float64"` | `"float32"` or `"float64"` |
| `remote_savepath` | path \| `None` | `None` | remote-side calculation directory |
| `loop_time` | `float` | `30` | seconds. polling interval for job completion |
| `stub` | `bool` | `False` | skip cluster submission, generate mock output instead |
| `md_steps_per_exchange` | `int` | `100` | steps. MD steps between replica-exchange attempts |
| `write_frequency` | `int` | `1` | steps. how often trajectory frames + per-step properties (energy, uncertainty) are written. Must be `<= n_steps_prod` |
| `swap_temperature` | `bool` | `True` | whether temperature exchanges are attempted |
| `rescale_momenta_on_swap` | `bool` | `True` | rescale momenta to the new temperature label on an accepted swap |
| `rescale_barostat_on_swap` | `bool` | `True` | also rescale NPT barostat DOFs on an accepted swap |

`model_paths` / `sampler_input` accepts either `["foundation"]` (MACE-MP foundation model) or a list of local paths to MACE model files. A single model path runs a standard simulation; **two or more** activates `MaceEnsembleModel` (shared message-passing body, per-member readout heads, ensemble-mean energy/forces + `energy_uncertainty`/`forces_uncertainty` attached to every output structure).

### Kwarg reference

All defaults below are `torch_sim`'s own fallback when the kwarg is omitted, expressed relative to `dt` (in native units, i.e. **after** the ps→native conversion already applied to `dt`).

| Kwarg | Used by | Meaning | Native units | Default if omitted |
|---|---|---|---|---|
| `tau` (step) | `nvt_vrescale` | V-Rescale thermostat relaxation time | time | `100 * dt` |
| `tau` (step) | `npt_crescale_*` | V-Rescale thermostat relaxation time | time | `100 * dt` |
| `gamma` (step) | `nvt_langevin` | particle friction | 1/time | `1 / (100 * dt)` |
| `tau` (init) | `nvt_nose_hoover` | Nosé–Hoover chain relaxation time | time | `100 * dt` |
| `chain_length` (init) | `nvt_nose_hoover`, `npt_nose_hoover_isotropic` | number of thermostats in the Nosé–Hoover chain | — | `3` |
| `chain_steps` (init) | `nvt_nose_hoover` | chain integration substeps | — | `3` |
| `chain_steps` (init) | `npt_nose_hoover_isotropic` | chain integration substeps | — | `2` |
| `sy_steps` (init) | `nvt_nose_hoover`, `npt_nose_hoover_isotropic` | Suzuki–Yoshida integration order (must be 1, 3, 5, or 7) | — | `3` |
| `alpha` (init) | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | particle friction | 1/time | `1 / (5 * dt)` |
| `cell_alpha` (init) | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | cell friction | 1/time | `1 / (30 * dt)` |
| `b_tau` (init) | `npt_langevin_isotropic`, `npt_langevin_anisotropic` | barostat time constant | time | `300 * dt` |
| `t_tau` (init) | `npt_nose_hoover_isotropic` | thermostat (temperature) relaxation time | time | `100 * dt` |
| `b_tau` (init) | `npt_nose_hoover_isotropic` | barostat (pressure) relaxation time | time | `1000 * dt` |
| `tau_p` (init) | `npt_crescale_*` | barostat relaxation time | time | `3 * dt` |
| `isothermal_compressibility` (init) | `npt_crescale_*` | isothermal compressibility | 1/pressure | `1e-6 bar⁻¹` (converted internally) |