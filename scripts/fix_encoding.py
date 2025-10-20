import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

# Fix markdown text
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        cell['source'] = [
            line.replace('Mean Field Game –', 'Mean Field Game –')
                .replace('emp�rica', 'empírica')
                .replace('configuração', 'configuração')
                .replace('Calibração', 'Calibração')
                .replace('Curva de oferta', 'Curva de oferta')
            for line in cell['source']
        ]

# Fix code text
replacements = {
    'Resultados ser�o': 'Resultados serão',
    'Calibra��o': 'Calibração',
    'Par�metros': 'Parâmetros',
    'M�tricas': 'Métricas',
    'Pre�o': 'Preço',
    'Arquivos gerados': 'Arquivos gerados',
    'Figuras': 'Figuras',
}
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            for old, new in replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell['source'] = new_source

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
