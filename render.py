"""Render figures."""

import platform
from importlib.metadata import version
import json

from pathlib import Path
import tempfile

from analyze import DEST, read_table
from experiments import figures as figs, readout

FTYPE = "png"
RENDER_SPEC = [
    (figs.convergence_figure, ("convergence",), f"convergence.{FTYPE}"),
    (figs.stability_figure, ("stability",), f"stability.{FTYPE}"),
    (figs.startup_figure, ("startup",), f"startup.{FTYPE}"),
    (figs.cost_figure, ("cost_trials", "matched_cost"), f"cost.{FTYPE}"),
    (figs.pulse_figure, ("pulse_profiles", "pulse_errors"), f"pulse.{FTYPE}"),
    (figs.positivity_figure, ("positivity",), f"positivity.{FTYPE}"),
    (figs.transfer_figure, ("transfer", "controller_cost"), f"transfer.{FTYPE}"),
    (figs.transient, (), f"startup.gif"),
]


def main():
    DEST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        tmp = Path(tmp)

        exp_data = {}
        for fig_factory, tables, fname in RENDER_SPEC:
            params = tuple(read_table(DEST / f"{table}.csv") for table in tables)
            exp_data.update(zip(tables, params))
            fig_factory(*params, tmp / fname)

        readout.write_readout(
            tmp / "README.md",
            **exp_data,
        )

        environment = dict(
            python=platform.python_version(),
            system=platform.system(),
            system_release=platform.release(),
            arch=platform.machine(),
            packages={
                p: version(p) for p in ("numpy", "scipy", "matplotlib", "Pillow")
            },
        )
        (tmp / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

        for path in tmp.iterdir():
            path.replace(DEST / path.name)

    print("Rendered all studies; results are in", DEST)


if __name__ == "__main__":
    main()
