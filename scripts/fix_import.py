import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['source'] = [line.replace('display, Image, Image', 'display, Image') for line in cell['source']]
nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
