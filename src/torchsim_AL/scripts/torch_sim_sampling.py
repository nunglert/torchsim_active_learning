"""
Torch-Sim parallel tempering sampling script.

Entry point for remote execution via SingleScriptTask. Reads a JSON input
file (path passed as sys.argv[1]), runs a parallel tempering simulation
using torch_sim with a MACE model, and writes the final replica structures
to ``out/samples.extxyz``.
"""

import os
import sys
import logging
import copy
from pathlib import Path
import json
from types import SimpleNamespace
import ase.io
import torch
import torch_sim as ts
from torchsim_AL.scripts.make_mace import _make_model
from torchsim_AL.scripts.parallel_tempering import parallel_tempering
from torchsim_AL.scripts.uncertainty_props import _build_prop_calculators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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

with open(INPUTFILE) as f:
    logger.info("Reading sampler config from %s", INPUTFILE)
    data = json.load(f)
    input_config = SimpleNamespace(**data)

# Resolve dtype and device
dtype_map = {"float32": torch.float32, "float64": torch.float64}
dtype = dtype_map[input_config.dtype]
device = torch.device(input_config.device)

# Load founder structure
at = ase.io.read(calcpath / input_config.founder)
logger.info("Loading founder structure from %s", input_config.founder)

# Load MACE model(s)
if len(input_config.model_paths) > 1:
    # Ensemble mode — pass list of paths; MaceEnsembleModel handles loading
    logger.info(
        "Loading %d-member MACE ensemble: %s",
        len(input_config.model_paths),
        input_config.model_paths,
        )
    mace = [calcpath / p for p in input_config.model_paths]
elif input_config.model_paths[0] == "foundation":
    logger.info("Loading MACE-MP foundation model")
    from mace.calculators.foundations_models import mace_mp
    mace = mace_mp(model="small", return_raw_model=True)
else:
    model_file = calcpath / input_config.model_paths[0]
    logger.info("Loading single MACE model from %s", model_file)
    mace = torch.load(model_file, map_location="cpu")
stress_conditional = input_config.integrator_name.startswith("npt")
model = _make_model(mace=mace, device=device, dtype=dtype, compute_forces=True, compute_stress=stress_conditional)

#make uncertainty props
n_batch=len(input_config.temperatures)
prop_calculators = _build_prop_calculators(model, frequency=input_config.write_frequency)
reporter_args = dict(
    filenames=[calcpath / f"trajectory_{i}.h5md" for i in range(n_batch)],
    state_frequency=input_config.write_frequency,
    prop_calculators=prop_calculators,
    )

#Prep input formats for torchsim
replica_atoms = [copy.deepcopy(at) for _ in range(n_batch)]

# Run simulation
common_kwargs = dict(
    replica_atoms=replica_atoms,
    model=model,
    integrator_name=input_config.integrator_name,
    temperatures=input_config.temperatures,
    dt=input_config.dt,
    init_kwargs=input_config.init_kwargs,
    step_kwargs=input_config.step_kwargs,
    n_total_steps=input_config.n_steps_pt,
    md_steps_per_exchange=input_config.md_steps_per_exchange,
    seed=input_config.seed,
    external_pressure=input_config.external_pressure,
    swap_temperature=input_config.swap_temperature,
    rescale_momenta_on_swap=input_config.rescale_momenta_on_swap,
    rescale_barostat_on_swap=input_config.rescale_barostat_on_swap,
    dtype=dtype,
    device=device
    )

logger.info("Starting parallel tempering: n_total_steps=%d, md_steps_per_exchange=%d", input_config.n_steps_pt, input_config.md_steps_per_exchange)
state, temperatures, P, swap_log = parallel_tempering(**common_kwargs, reporter_args=reporter_args)
logger.info(
    "Parallel tempering finished: accept_rate=%.3f (%d/%d attempted)",
    swap_log["info"]["accept_rate"],
    swap_log["info"]["accepted"],
    swap_log["info"]["attempted"],
    )

swap_log_path = calcpath / "pt_swaps.json"
with swap_log_path.open("w") as f:
    json.dump(swap_log, f)
logger.info("Wrote swap log to %s", swap_log_path)