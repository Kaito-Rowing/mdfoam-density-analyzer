from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


try:
    from mdfoam_analyzer.gui import main
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(
        f"Missing dependency: {missing}\n"
        "Install the GUI dependencies with:\n"
        "  python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
