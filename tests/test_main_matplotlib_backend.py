import os
import subprocess
import sys
from pathlib import Path


def test_main_forces_headless_matplotlib_backend():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["MPLBACKEND"] = "module://backend_interagg"
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = (
        "import os; "
        "import main; "
        "import matplotlib; "
        "assert os.environ['MPLBACKEND'] == 'Agg'; "
        "assert matplotlib.get_backend().lower() == 'agg'"
    )

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=30,
    )
