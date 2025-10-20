import json
from pathlib import Path
nb = json.loads(Path('notebooks/mfg_pipeline.ipynb').read_text())
print(nb['cells'][0]['source'])
print(nb['cells'][6]['source'])
