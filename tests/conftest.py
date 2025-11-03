import sys
from pathlib import Path


def pytest_load_initial_conftests(args):
    # ensure repository root is on sys.path so tests can import `src` package
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
