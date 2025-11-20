import os
from pathlib import Path

from astune import launcher


def test_launcher_dry_run_appworld_linear_verl(tmp_path):
    # Use the provided config path
    conf_path = "tutorial/example_appworld/appworld.yaml"
    assert Path(conf_path).exists(), "Configuration file should exist for the test"

    # Run in dry-run mode to avoid heavy training
    result = launcher.run_for_test(
        conf=conf_path, backbone="verl", with_appworld=False, dry_run=True
    )

    # Validate returned structure
    assert "yaml" in result and result["yaml"], "Expect backup yaml path in result"
    assert Path(result["yaml"]).exists(), "Backup YAML file should exist after preparation"
    assert (
        result.get("exp_base") and Path(result["exp_base"]).exists()
    ), "Experiment base folder should exist"

    # Ensure launcher_record structure was created
    exp_name = result.get("exp_name")
    assert exp_name, "Experiment name should be derived"
    record_dir = Path("launcher_record") / exp_name
    assert record_dir.exists(), "launcher_record directory for experiment should exist"

    # Backup YAML should have backbone set to verl
    import yaml

    with open(result["yaml"], "r") as f:
        data = yaml.safe_load(f)
    assert data.get("astune", {}).get("backbone") == "verl", "Backbone should be set in backup yaml"

    # Dry run should NOT have produced training artifacts (e.g., checkpoints)
    ckpt_dir = Path("checkpoints")
    # We only assert that no new subdir for exp_name appears (optional)
    if ckpt_dir.exists():
        assert not any(
            p.name.startswith(exp_name) for p in ckpt_dir.iterdir()
        ), "Dry run should not create checkpoints"


def test_launcher_multiple_runs_idempotent(tmp_path):
    conf_path = "tutorial/example_appworld/appworld.yaml"
    result1 = launcher.run_for_test(conf=conf_path, backbone="verl", dry_run=True)
    result2 = launcher.run_for_test(conf=conf_path, backbone="verl", dry_run=True)
    # Should reuse same experiment name and keep yaml present
    assert result1["exp_name"] == result2["exp_name"]
    assert Path(result2["yaml"]).exists()
