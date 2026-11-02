#!/usr/bin/env python3
"""Git-фильтр: убирает выводы из ноутбука при коммите, рабочий файл не меняется. Только стандартная библиотека."""
import json
import sys

nb = json.load(sys.stdin)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
json.dump(nb, sys.stdout, ensure_ascii=False, indent=1)
sys.stdout.write("\n")
