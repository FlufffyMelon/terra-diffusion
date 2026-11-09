#!/usr/bin/env python3
"""Git filter: strips outputs from the notebook on commit, the working file is not changed. Standard library only."""
import json
import sys

nb = json.load(sys.stdin)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
json.dump(nb, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write("\n")
