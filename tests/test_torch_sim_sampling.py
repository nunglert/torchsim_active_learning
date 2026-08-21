import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import torchsim_AL
from torchsim_AL import sampler_spec

SCRIPT_PATH = (
    Path(torchsim_AL.__file__).parent / "scripts" / "torch_sim_sampling.py"
)


def _base_kwargs(**overrides):
    kwargs = dict(
        founder="founder.extxyz",
        integrator_name="nvt_langevin",
        dt=0.001,
        init_kwargs={},
        step_kwargs={"gamma": 1.0},
        temperatures=[300.0, 400.0, 500.0],
        model_paths=["foundation"],
    )
    kwargs.update(overrides)
    return kwargs


def _attrs_read_from_input_config(script_path: Path) -> set[str]:
    """Every ``input_config.X`` attribute the script reads."""
    tree = ast.parse(script_path.read_text())
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "input_config"
    }


def test_spec_is_json_round_trippable(tmp_path):
    """validate_spec's output must survive json.dumps -> json.load unchanged."""
    spec = sampler_spec.validate_spec(**_base_kwargs())

    input_file = tmp_path / "input_torchsim.json"
    input_file.write_text(json.dumps(spec, indent=2))

    reloaded = json.loads(input_file.read_text())
    assert reloaded == spec

    input_config = SimpleNamespace(**reloaded)
    assert vars(input_config) == spec


def test_script_reads_only_keys_the_spec_provides():
    """No attribute access in the script may be missing from the spec dict."""
    spec = sampler_spec.validate_spec(**_base_kwargs())
    missing = _attrs_read_from_input_config(SCRIPT_PATH) - set(spec)
    assert not missing, (
        f"torch_sim_sampling.py reads {sorted(missing)} off input_config, but "
        f"validate_spec does not emit them. Available: {sorted(spec)}"
    )


def test_spec_has_no_keys_the_script_ignores():
    """The reverse direction: flags fields that silently do nothing."""
    spec = sampler_spec.validate_spec(**_base_kwargs())
    unused = set(spec) - _attrs_read_from_input_config(SCRIPT_PATH)
    assert not unused, (
        f"validate_spec emits {sorted(unused)}, which torch_sim_sampling.py "
        "never reads. Either wire them up or drop them from the spec."
    )


@pytest.mark.slow
def test_script_runs_end_to_end(tmp_path):
    """Execute the real script on a tiny CPU job.

    Needs mace-torch installed (the script imports MACE unconditionally) plus
    the MACE-MP foundation model, which the script itself pulls for
    ``model_paths=["foundation"]``. Warming the download here rather than in the
    subprocess keeps a missing model a clean skip instead of an opaque non-zero
    exit; mace caches it under ``~/.cache/mace`` after the first fetch.
    """
    pytest.importorskip("mace", reason="mace-torch not installed")

    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    try:
        download_mace_mp_checkpoint("small")
    except Exception as exc:  # no cached copy and no network
        pytest.skip(f"MACE-MP foundation model unavailable: {exc}")

    founder_src = Path(__file__).parent.parent / "al_example" / "inputs" / (
        "Cu2S_hexagonal.extxyz"
    )
    if not founder_src.exists():
        pytest.skip(f"founder structure not available at {founder_src}")

    shutil.copy(founder_src, tmp_path / founder_src.name)

    spec = sampler_spec.validate_spec(
        **_base_kwargs(
            founder=founder_src.name,
            model_paths=["foundation"],
            temperatures=[300.0, 400.0],
            n_steps_equil=0,
            n_steps_prod=4,
            md_steps_per_exchange=2,
            write_frequency=2,
            device="cpu",
        )
    )
    input_file = tmp_path / "input_torchsim.json"
    input_file.write_text(json.dumps(spec, indent=2))

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(input_file)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (
        f"script exited {result.returncode}\nSTDERR:\n{result.stderr}"
    )

    for i in range(len(spec["temperatures"])):
        assert (tmp_path / f"trajectory_{i}.h5md").exists()

    swap_log = json.loads((tmp_path / "pt_swaps.json").read_text())
    assert "accept_rate" in swap_log["info"]
    assert swap_log["info"]["attempted"] >= swap_log["info"]["accepted"]