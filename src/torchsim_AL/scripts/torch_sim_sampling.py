"""
Torch-Sim parallel tempering sampling script.

Entry point for remote execution via SingleScriptTask. Reads a JSON input
file (path passed as sys.argv[1]), runs a parallel tempering simulation
using torch_sim with a MACE model, and writes the final replica structures
to ``out/samples.extxyz``.
"""

import os
import sys
from pathlib import Path
import json
from types import SimpleNamespace
import ase.io
import torch
from torchsim_AL.scripts.make_mace import _make_model
from torchsim_AL.scripts.parallel_tempering import parallel_tempering
from torchsim_AL.scripts.uncertainty_props import _build_prop_calculators

# Redirect temp files to ~/tmp unconditionally.  Some GPU nodes have a
# non-writable /tmp which causes NVIDIA's NVRTC compiler (used by warp for
# CUDA kernel compilation) to fail with "cannot open temporary file
# /tmp/edg20_*".  NVRTC's EDG frontend reads TMPDIR to locate its temp
# directory, so setting it here before any CUDA imports fixes the issue.
_tmp_fallback = Path.home() / "tmp"
_tmp_fallback.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_tmp_fallback)
os.environ["TEMP"] = str(_tmp_fallback)
os.environ["TMP"] = str(_tmp_fallback)

INPUTFILE = Path(sys.argv[1])
calcpath = INPUTFILE.parent

with open("config.json") as f:
    data = json.load(f)
    input_config = SimpleNamespace(**data)

# Resolve dtype
dtype_map = {"float32": torch.float32, "float64": torch.float64}
dtype = dtype_map[input_config.dtype]

# Load founder structure
at = ase.io.read(calcpath / input_config.founder)

# Load MACE model(s)
if len(input_config.model_paths) > 1:
    # Ensemble mode — pass list of paths; MaceEnsembleModel handles loading
    mace = [calcpath / p for p in input_config.model_paths]
elif input_config.model_paths[0] == "foundation":
    from mace.calculators.foundations_models import mace_mp
    mace = mace_mp(model="small", return_raw_model=True)
else:
    mace = torch.load(
        calcpath / input_config.model_paths[0], map_location="cpu"
    )
model = _make_model(mace=mace, device=input_config.device, dtype=dtype, compute_forces=True, compute_stress=False)

#make uncertainty props
n_batch=len(input_config.temperatures)
prop_calculators = _build_prop_calculators(model)
reporter_args = dict(
    filenames=[calcpath / f"trajectory_{i}.h5md" for i in range(n_batch)],
    state_frequency=input_config.write_frequency,
    prop_calculators=prop_calculators,
)

# Run simulation
common_kwargs = dict(
    at=at,
    calcpath=calcpath,
    temperatures=input_config.temperatures,
    mace=mace,
    dt=input_config.dt,
    external_pressure=input_config.external_pressure,
    n_steps_npt=input_config.n_steps_npt,
    n_steps_pt=input_config.n_steps_pt,
    seed=input_config.seed,
    device=torch.device(input_config.device),
    dtype=dtype,
    md_steps_per_exchange=input_config.md_steps_per_exchange,
    init_kwargs=input_config.init_kwargs,
    step_kwargs=input_config.step_kwargs,
)

state, temperatures, P, swap_log = parallel_tempering(**common_kwargs, reporter_args=reporter_args)

#write swap_log
with Path(calcpath / "pt_swaps.json").open("w") as f:
    json.dump(swap_log, f)