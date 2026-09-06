"""Source checkout entry point; installed users can run sproutko-generate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sproutko.cli import main

if __name__ == "__main__":
    main()
