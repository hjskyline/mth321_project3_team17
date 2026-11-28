"""Run verification, regenerate CSV evidence and render figures in one command."""

import subprocess, sys
from pathlib import Path
import analyze, render

ROOT = Path(__file__).resolve().parent


def main():
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=True,
    )
    analyze.main()
    render.main()


if __name__ == "__main__":
    main()
