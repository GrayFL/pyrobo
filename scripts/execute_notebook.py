"""Execute the example with this Python interpreter, irrespective of kernelspec paths.

Run from any directory: python scripts/execute_notebook.py
The executed notebook is saved in place, including partial output on failure.
"""
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "navigation_2d.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, timeout=180, km=manager,
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells: {path}")


if __name__ == "__main__":
    main()

