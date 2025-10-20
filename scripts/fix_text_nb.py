import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

replacements = {
    'CalibraÃ§Ã£o heurÃ­stica': 'Calibração heurística',
    'Curva de oferta empÃ­rica': 'Curva de oferta empírica',
    'CalibraÃ§Ã£o': 'Calibração',
    'heurÃ­stica': 'heurística',
    'empÃ­rica': 'empírica',
    'Resultados serÃ£o': 'Resultados serão',
    'ParÃ¢metros': 'Parâmetros',
    'MÃ©tricas': 'Métricas',
    'PreÃ§o': 'Preço'
}

for cell in nb['cells']:
    if cell['cell_type'] in {'markdown', 'code'}:
        new_source = []
        for line in cell.get('source', []):
            for old, new in replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell['source'] = new_source

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
