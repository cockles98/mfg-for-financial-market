import json
from pathlib import Path
nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f'# Cell {i}')
        print(''.join(cell['source']))
        print()
