import json
import unicodedata
from pathlib import Path

normalize = lambda s: ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)).lower()
nb = json.loads(Path('notebooks/mfg_pipeline.ipynb').read_text())
for i, cell in enumerate(nb['cells']):
    if cell['cell_type']=='markdown':
        txt = ''.join(cell['source'])
        print(i, normalize(txt).split('\n')[0])
