import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

md_map = {
    'Mean Field Game � Data-driven Pipeline': 'Mean Field Game – Data-driven Pipeline',
    'Calibra��o heur�stica': 'Calibração heurística',
    'Curva de oferta emp�rica': 'Curva de oferta empírica',
    'Relat�rios': 'Relatórios',
    's�o': 'são'
}
code_map = {
    'Resultados ser�o': 'Resultados serão',
    'Calibra��o': 'Calibração',
    'M�tricas': 'Métricas',
    'Par�metros': 'Parâmetros',
    'Pre�o': 'Preço'
}

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        cell['source'] = [line.replace(old, new) for line in cell['source'] for old, new in md_map.items() if old in line] or cell['source']
        new_src = []
        for line in cell['source']:
            for old, new in md_map.items():
                line = line.replace(old, new)
            new_src.append(line)
        cell['source'] = new_src
    elif cell['cell_type'] == 'code':
        new_src = []
        for line in cell['source']:
            for old, new in code_map.items():
                line = line.replace(old, new)
            new_src.append(line)
        cell['source'] = new_src

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')