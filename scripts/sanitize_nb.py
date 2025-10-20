import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

for cell in nb['cells']:
    if cell['cell_type'] in {'code','markdown'}:
        sanitized = []
        for line in cell.get('source', []):
            if isinstance(line, str):
                sanitized.append(line.encode('latin-1', errors='ignore').decode('latin-1'))
            else:
                sanitized.append(line)
        cell['source'] = sanitized

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
