import json
from pathlib import Path
nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    preview = src.strip().split('\n')[0][:80] if src else ''
    print(i, cell['cell_type'], '->', preview)
