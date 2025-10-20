import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

replacements = {
    '# Mean Field Game  Data-driven Pipeline': '# Mean Field Game – Data-driven Pipeline',
    '## 2.1 Curva de oferta emprica\n': '## 2.1 Curva de oferta empírica\n',
    'Curva de oferta emprica': 'Curva de oferta empírica',
}

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            for old, new in replacements.items():
                line = line.replace(old, new)
            # also fix any stray  or etc
            line = line.replace('\u0012', '–')
            new_source.append(line)
        cell['source'] = new_source

# Also ensure code prints are proper ASCII/unicode
code_replacements = {
    'Ã©': 'é',
    'Ã£': 'ã',
    'Ãº': 'ú',
    'Ã¡': 'á',
    'Ã³': 'ó',
    'Ãª': 'ê',
    'Ã§': 'ç',
    'Ãº': 'ú',
    'Ã­': 'í',
    'Ã¹': 'ù',
}
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            for old, new in code_replacements.items():
                line = line.replace(old, new)
            new_source.append(line)
        cell['source'] = new_source

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
