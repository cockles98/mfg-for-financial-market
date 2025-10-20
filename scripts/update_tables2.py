import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
cells = nb['cells']

# Ensure import includes Image
for cell in cells:
    if cell['cell_type'] == 'code' and any('from IPython.display import display' in line for line in cell.get('source', [])):
        cell['source'] = [line.replace('from IPython.display import display', 'from IPython.display import display, Image') for line in cell['source']]
        break

# Insert supply section if missing
if not any(cell['cell_type']=='markdown' and 'Curva de oferta' in ''.join(cell.get('source', [])) for cell in cells):
    supply_md = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 2.1 Curva de oferta empírica\n', 'Quantis de volume e spread usados no clearing.\n']
    }
    supply_code = {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [
            "supply_path = DATA_PROCESSED / 'supply_curve.csv'\n",
            "if supply_path.exists():\n",
            "    supply_df = pd.read_csv(supply_path)\n",
            "    print('Curva de oferta por quantil:')\n",
            "    display(supply_df)\n",
            "else:\n",
            "    print('Arquivo supply_curve.csv não encontrado; execute a calibração empírica.')\n",
        ]
    }
    calib_md_idx = next(i for i, cell in enumerate(cells) if cell['cell_type']=='markdown' and cell.get('source') and cell['source'][0].strip().startswith('## 2.'))
    cells.insert(calib_md_idx + 1, supply_md)
    cells.insert(calib_md_idx + 2, supply_code)

# Helper to find code cell containing keyword

def find_cell(keyword):
    for cell in cells:
        if cell['cell_type']=='code' and keyword in ''.join(cell.get('source', [])):
            return cell
    raise ValueError(f'Cell with keyword {keyword!r} not found')

# Calibration cell -> display DF
calib_cell = find_cell('calibration_summary =')
if "calibration_df = pd.DataFrame([calibration_summary])\n" not in calib_cell['source']:
    calib_cell['source'].append("calibration_df = pd.DataFrame([calibration_summary])\n")
    calib_cell['source'].append("display(calibration_df)\n")

# Baseline params cell -> display
baseline_cell = find_cell("baseline_cfg['params']['nu']")
if "display(pd.DataFrame([baseline_cfg['params']]))\n" not in baseline_cell['source']:
    baseline_cell['source'].append("display(pd.DataFrame([baseline_cfg['params']]))\n")

# Solver metrics -> display
solver_cell = find_cell('run_metrics = {')
if "display(pd.DataFrame([run_metrics]))\n" not in solver_cell['source']:
    solver_cell['source'].append("display(pd.DataFrame([run_metrics]))\n")

# Summary -> display
summary_cell = find_cell('summary_display = {')
if "summary_df = pd.DataFrame([summary_display])\n" not in summary_cell['source']:
    summary_cell['source'].append("summary_df = pd.DataFrame([summary_display])\n")
    summary_cell['source'].append("display(summary_df)\n")

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
