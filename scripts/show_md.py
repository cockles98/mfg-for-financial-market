import json
from pathlib import Path
nb = json.loads(Path('notebooks/mfg_pipeline.ipynb').read_text())
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type']=='markdown':
        print(idx, cell['source'])
