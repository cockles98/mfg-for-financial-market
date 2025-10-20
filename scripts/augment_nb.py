import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

code_indices = {2,4,6,8,10,12,14,16,18,20}
for idx, cell in enumerate(nb['cells']):
    if idx == 2 and cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'from IPython.display import display' not in src:
            cell['source'].append('from IPython.display import display\n')
            cell['source'].append('import pprint\n')
    if idx == 4 and cell['cell_type'] == 'code':
        cell['source'].append("print(f'Resultados serão salvos em: {REPORT_DIR}')\n")
    if idx == 6 and cell['cell_type'] == 'code':
        cell['source'].append("print('Calibração heurística:')\n")
        cell['source'].append("print(json.dumps(calibration_summary, indent=2))\n")
    if idx == 8 and cell['cell_type'] == 'code':
        cell['source'].append("print('Parâmetros atuais do baseline:')\n")
        cell['source'].append("print(json.dumps(baseline_cfg['params'], indent=2))\n")
    if idx == 10 and cell['cell_type'] == 'code':
        cell['source'].append("print('Grid configurado:')\n")
        cell['source'].append("print(grid)\n")
        cell['source'].append("print('Parâmetros HFT:')\n")
        cell['source'].append("print(params)\n")
    if idx == 12 and cell['cell_type'] == 'code':
        cell['source'].append("print('Métricas da simulação:')\n")
        cell['source'].append("for k, v in run_metrics.items():\n    print(f'  {k}: {v}')\n")
    if idx == 14 and cell['cell_type'] == 'code':
        cell['source'].append("if price_results is not None:\n    print(f'Preço médio: {run_metrics.get('price_mean'):.4f} | desvio: {run_metrics.get('price_std'):.4f}')\nelse:\n    print('Clearing de preço desativado.')\n")
    if idx == 16 and cell['cell_type'] == 'code':
        cell['source'].append("print('Arquivos gerados:')\n")
        cell['source'].append("for name in sorted(p.name for p in REPORT_DIR.iterdir()):\n    print('  -', name)\n")
    if idx == 18 and cell['cell_type'] == 'code':
        cell['source'].append("print('Figuras atualizadas em', REPORT_DIR)\n")
    if idx == 20 and cell['cell_type'] == 'code':
        cell['source'].append("print('Resumo final:')\n")
        cell['source'].append("for k, v in summary_display.items():\n    print(f'  {k}: {v}')\n")

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
