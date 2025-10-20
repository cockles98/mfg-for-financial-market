import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
cells = nb['cells']

# ensure import cell includes Image
for cell in cells:
    if cell['cell_type'] == 'code' and any('from IPython.display import display' in line for line in cell.get('source', [])):
        cell['source'] = [line.replace('from IPython.display import display', 'from IPython.display import display, Image') for line in cell['source']]
        break

# insert supply markdown/code if not present
has_supply = any(cell['cell_type']=='markdown' and cell['source'] and cell['source'][0].strip().startswith('## 2.1') for cell in cells)
if not has_supply:
    supply_md = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 2.1 Curva de oferta empírica\n', 'Mostra quantis de volume e spread que alimentam o clearing.\n']
    }
    supply_code = {
        'cell_type': 'code',
        'metadata': {},
        'outputs': [],
        'execution_count': None,
        'source': [
            "supply_path = DATA_PROCESSED / 'supply_curve.csv'\n",
            "if supply_path.exists():\n",
            "    supply_df = pd.read_csv(supply_path)\n",
            "    print('Curva de oferta por quantil:')\n",
            "    display(supply_df)\n",
            "else:\n",
            "    print('Arquivo supply_curve.csv não encontrado; execute a calibração empírica.')\n"
        ]
    }
    # insert after calibration cell (first code containing 'Calibração heurística')
    calib_idx = next(i for i, cell in enumerate(cells) if cell['cell_type']=='markdown' and 'Calibração heurística' in ''.join(cell.get('source', [])))
    cells.insert(calib_idx+1, supply_md)
    cells.insert(calib_idx+2, supply_code)

# helper to find code cell by keyword

def find_code(keyword):
    for i, cell in enumerate(cells):
        if cell['cell_type']=='code' and any(keyword in line for line in cell.get('source', [])):
            return i, cell
    raise ValueError(f'Code cell with keyword {keyword} not found')

# calibration cell -> display DF
idx, calib_cell = find_code("calibration_summary =")
if "display(pd.DataFrame([calibration_summary]))\n" not in calib_cell['source']:
    calib_cell['source'].append("calibration_df = pd.DataFrame([calibration_summary])\n")
    calib_cell['source'].append("display(calibration_df)\n")

# baseline cell display params
idx, base_cell = find_code("baseline_cfg['params']['nu']")
if "display(pd.DataFrame([baseline_cfg['params']]))\n" not in base_cell['source']:
    base_cell['source'].append("display(pd.DataFrame([baseline_cfg['params']]))\n")

# solver metrics cell -> display run_metrics DF
idx, solver_cell = find_code("run_metrics = {")
if "display(pd.DataFrame([run_metrics]))\n" not in solver_cell['source']:
    solver_cell['source'].append("display(pd.DataFrame([run_metrics]))\n")

# summary cell -> display summary
idx, summary_cell = find_code("summary_display =")
if "display(pd.DataFrame([summary_display]))\n" not in summary_cell['source']:
    summary_cell['source'].append("display(pd.DataFrame([summary_display]))\n")

# ensure supply code uses display (already done when inserted)

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
