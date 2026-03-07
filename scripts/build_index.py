from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Prevent noisy MKL threading warnings on Windows for MiniBatchKMeans.
os.environ.setdefault("OMP_NUM_THREADS", "4")

from app.engine import build_artifacts


if __name__ == "__main__":
    report = build_artifacts()
    print("Build complete:")
    for key, value in report.items():
        print(f"  {key}: {value}")
