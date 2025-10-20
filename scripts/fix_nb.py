import nbformat
from pathlib import Path
nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = nbformat.read(nb_path, as_version=4)
replacements = {
    'Calibra�o': 'Calibração',
    'emp�rica': 'empírica',
    'grA�ficos': 'gráficos',
    'Resultados serão': 'Resultados serão',
    'Parâmetros': 'Parâmetros',
    'Métricas': 'Métricas',
    'Preço': 'Preço',
}
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        for old, new in list(replacements.items()):
            cell.source = cell.source.replace(old, new)
    elif cell.cell_type == 'code':
        for old, new in list(replacements.items()):
            cell.source = cell.source.replace(old, new)
nbformat.write(nb, nb_path)
