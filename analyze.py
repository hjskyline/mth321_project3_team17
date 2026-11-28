"""Generate CSV evidence."""

import csv
from pathlib import Path
import tempfile

from experiments import studies, transfer as txfer

DEST = Path(__file__).resolve().parent / "results"


def infer_type(value: str):
    if value.isdecimal():
        return int(value)
    try:
        return float(value)
    except ValueError as e:
        return value


def read_table(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return [{k: infer_type(v) for k, v in r.items()} for r in reader]


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    DEST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DEST) as tmp:
        tmp = Path(tmp)
        for name, rows in {
            "convergence": studies.convergence(),
            "stability": studies.stability(),
            "startup": studies.startup(),
            ("cost_trials", "matched_cost"): studies.cost(),
            "transfer": txfer.transfer(),
            ("pulse_profiles", "pulse_errors"): txfer.pulse(),
            "positivity": txfer.positivity(),
            "controller_cost": txfer.controller_cost(),
        }.items():
            if isinstance(name, str):
                write_table(tmp / f"{name}.csv", rows)
            else:
                for n, r in zip(name, rows):
                    write_table(tmp / f"{n}.csv", r)

        for path in tmp.iterdir():
            path.replace(DEST / path.name)

    print("Wrote all tables into", DEST)


if __name__ == "__main__":
    main()
