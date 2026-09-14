"""Static contract check for the independent inference-timing launchers."""

from ast import parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    script = ROOT / "src/timebench/scripts/time_inference.py"
    runner = ROOT / "src/slurm/time_inference.sh"
    dgx = ROOT / "slurm/dgx/time_inference.slurm"
    selena = ROOT / "slurm/selena/time_inference.slurm"

    parse(script.read_text(encoding="utf-8"), filename=str(script))
    runner_text = runner.read_text(encoding="utf-8")
    assert "${1:-SG_Weather/D}" in runner_text
    assert "${2:-short}" in runner_text
    assert "${3:-30}" in runner_text
    assert "timebench.scripts.time_inference" in runner_text
    for front in (dgx, selena):
        assert front.is_file()
        assert "src/slurm/time_inference.sh" in front.read_text(encoding="utf-8")

    assert not (ROOT / "slurm/selena/time_inference_selena.slurm").exists()
    print("time inference contract passed")


if __name__ == "__main__":
    main()
