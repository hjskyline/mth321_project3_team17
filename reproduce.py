"""Run verification, regenerate CSV evidence and render figures in one command."""
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

from experiments import figures
from experiments import studies
from experiments import transfer
from experiments.readout import write_readout


ROOT = Path(__file__).resolve().parent


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                   cwd=ROOT, check=True)
    dest = ROOT / "results"
    dest.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        out = Path(tmp)
        convergence = studies.convergence()
        stability = studies.stability()
        startup = studies.startup()
        cost, matched = studies.cost()
        validation = transfer.transfer()
        points, pulse = transfer.pulse()
        spike = transfer.positivity()
        overhead = transfer.controller_cost()
        for name, rows in (("convergence", convergence), ("stability", stability),
                           ("startup", startup), ("cost_trials", cost), ("matched_cost", matched),
                           ("transfer", validation), ("pulse_profiles", points), ("pulse_errors", pulse),
                           ("positivity", spike), ("controller_cost", overhead)):
            write_table(out / f"{name}.csv", rows)
        figures.convergence_figure(convergence, out / "convergence.png")
        figures.stability_figure(stability, out / "stability.png")
        figures.startup_figure(startup, out / "startup.png")
        figures.cost_figure(cost, matched, out / "cost.png")
        figures.pulse_figure(points, pulse, out / "pulse.png")
        figures.positivity_figure(spike, out / "positivity.png")
        figures.transfer_figure(validation, overhead, out / "transfer.png")
        write_readout(out/"README.md", pulse, spike, validation, overhead, matched, convergence)
        figures.transient(out / "startup.gif")
        environment = dict(python=platform.python_version(), system=platform.system(),
                           system_release=platform.release(), arch=platform.machine(),
                           packages={p:version(p) for p in ("numpy","scipy","matplotlib","Pillow")})
        (out/"environment.json").write_text(json.dumps(environment, indent=2)+"\n")
        for path in out.iterdir():
            path.replace(dest / path.name)
    print("Verified all studies; results are in", dest)


if __name__ == "__main__":
    main()
