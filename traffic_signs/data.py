from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def resolved_data_yaml(out=None):
    """Write a copy of configs/signs.yaml with the absolute dataset path for this machine."""
    cfg = yaml.safe_load((ROOT / "configs" / "signs.yaml").read_text())
    cfg["path"] = str((ROOT / "database" / "car").resolve())
    out = Path(out or ROOT / "runs" / "_signs_resolved.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)
