import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

for idx, cell in enumerate(nb['cells']):
    if idx == 12 and cell['cell_type'] == 'code':
        cell['source'] = [line.replace("print(f'  {k}: {v}')", "print(f\"  {k}: {v}\")") for line in cell['source']]
    if idx == 14 and cell['cell_type'] == 'code':
        cell['source'] = [
            line.replace("print(f\"Preço médio: {run_metrics.get('price_mean', 0.0):.4f} | desvio: {run_metrics.get('price_std', 0.0):.4f}\")",
                         "print(f\"Preço médio: {run_metrics.get('price_mean', 0.0):.4f} | desvio: {run_metrics.get('price_std', 0.0):.4f}\")")
            for line in cell['source']
        ]
    if idx == 16 and cell['cell_type'] == 'code':
        cell['source'] = [line.replace("print('  -', name)", "print('  -', name)") for line in cell['source']]
    if idx == 20 and cell['cell_type'] == 'code':
        cell['source'] = [line.replace("print(f'  {k}: {v}')", "print(f\"  {k}: {v}\")") for line in cell['source']]

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
