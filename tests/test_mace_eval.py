import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones

from remote_handler.exceptions import ScriptInputError
from remote_handler.processing.registry import REGISTRY, validate_args

from torchsim_AL.processors import mace_eval
from torchsim_AL.processors.mace_eval import MaceEvalArgs, MaceEvalProcessor


def _scaled_lj(scale):
    class ScaledLJ(LennardJones):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results = {k: v * scale for k, v in self.results.items()}

    return ScaledLJ(sigma=2.3, epsilon=0.01, rc=5.0)


@pytest.fixture
def atoms_list():
    frames = []
    rng = np.random.default_rng(0)
    for _ in range(3):
        at = bulk("Cu", "fcc", a=3.6, cubic=True).repeat(2)
        at.rattle(0.05, seed=int(rng.integers(1000)))
        at.new_array("REF_forces", rng.normal(size=(len(at), 3)))
        frames.append(at)
    frames[-1].arrays.pop("REF_forces")
    return frames


@pytest.fixture
def fake_models(monkeypatch):
    scales = {"a.pkl": 1.0, "b.pkl": 3.0, "c.pkl": 2.0}
    monkeypatch.setattr(
        mace_eval,
        "_load_calculator",
        lambda path, device, dtype: _scaled_lj(scales[path]),
    )
    return scales


def test_registered():
    assert REGISTRY["mace_eval"] is mace_eval.run


def test_run_annotates_committee_stats(atoms_list, fake_models):
    args = {
        "committees": {"it0": ["a.pkl", "b.pkl"], "it1": ["c.pkl"]},
        "device": "cpu",
        "write_forces": True,
    }
    base = _scaled_lj(1.0)
    ref_e, ref_f = [], []
    for at in atoms_list:
        c = at.copy()
        c.calc = base
        ref_e.append(c.get_potential_energy())
        ref_f.append(c.get_forces())

    out = mace_eval.run(atoms_list, args, calcpath=None)

    for i, at in enumerate(out):
        assert at.info["it0_energy_m0"] == pytest.approx(ref_e[i])
        assert at.info["it0_energy_m1"] == pytest.approx(3 * ref_e[i])
        assert at.info["it0_energy"] == pytest.approx(2 * ref_e[i])
        assert at.info["it0_energy_std"] == pytest.approx(abs(ref_e[i]))
        assert at.info["it1_energy"] == pytest.approx(2 * ref_e[i])
        assert at.info["it1_energy_std"] == pytest.approx(0.0)
        np.testing.assert_allclose(at.arrays["it0_forces"], 2 * ref_f[i])
        if "REF_forces" in at.arrays:
            ref = at.arrays["REF_forces"]
            assert at.info["it0_forces_sse"] == pytest.approx(
                ((2 * ref_f[i] - ref) ** 2).sum()
            )
            assert at.info["it0_forces_sse_m1"] == pytest.approx(
                ((3 * ref_f[i] - ref) ** 2).sum()
            )
        else:
            assert "it0_forces_sse" not in at.info


def test_processor_stages_models_and_builds_args(tmp_path):
    models = {"it0": [tmp_path / "m0.pkl", tmp_path / "m1.pkl"]}
    proc = MaceEvalProcessor(models, device="cuda")
    assert proc.shared_files() == [str(p) for p in models["it0"]]
    args = proc.build_args("/remote/dump")
    assert args["committees"] == {
        "it0": ["/remote/dump/m0.pkl", "/remote/dump/m1.pkl"]
    }
    validate_args("mace_eval", args)
    assert proc.registration_module == "torchsim_AL.processors.mace_eval"


def test_processor_rejects_duplicate_basenames(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        MaceEvalProcessor(
            {"a": [tmp_path / "x" / "m.pkl"], "b": [tmp_path / "y" / "m.pkl"]}
        )


def test_args_validation():
    with pytest.raises(ScriptInputError):
        MaceEvalArgs(committees={}).validate()
    with pytest.raises(ScriptInputError):
        MaceEvalArgs(committees={"a": []}).validate()
    with pytest.raises(ScriptInputError):
        MaceEvalArgs(committees={"a": ["m"]}, default_dtype="float16").validate()
