import importlib.util
import json
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "simulation.py"
    spec = importlib.util.spec_from_file_location("global_manifold_audit_simulation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_writes_relative_audit_outputs(tmp_path):
    module = load_module()

    result = module.main(n_simulations=5, seed=42, project_root=tmp_path)

    certification_path = tmp_path / "certification.json"
    results_path = tmp_path / "audit_results.csv"

    assert certification_path.exists()
    assert results_path.exists()

    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    assert certification["status"] == "MASS_VALIDATED"
    assert certification["n_simulations"] == 5
    assert len(result["results"]) == 5
