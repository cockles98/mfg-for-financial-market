import json
from pathlib import Path

nb = json.loads(Path('notebooks/mfg_pipeline.ipynb').read_text())
# Fix first two markdown cells
nb['cells'][0]['source'] = ['# Mean Field Game – Data-driven Pipeline\n', '\n', 'Este notebook calibra o modelo com dados da B3, executa o solver LQ e explica cada etapa de forma intuitiva.\n']
nb['cells'][1]['source'] = ['## Narrativa geral\n', "- **Dados**: usamos COTAHIST limpo para extrair volatilidade, retornos, spreads e volume.\n", "- **Calibração**: heurísticas ajustam `nu`, `phi`, `gamma_T`, `eta0`, `eta1`.\n", "- **Solver**: executamos o Picard com salvaguardas.\n", "- **Relatórios**: resultados são salvos em `notebooks_output/`.\n"]

# Fix other markdown cells
replace_map = {
    'Calibra��o': 'Calibração',
    'heur�stica': 'heurística',
    'Curva de oferta emp�rica': 'Curva de oferta empírica',
    'RelatA3rios': 'Relatórios',
    'sA�o': 'são'
}
for idx in [3,5,6,9,11,13,15,17,19,21,23]:
    src = nb['cells'][idx]['source']
    new_src = []
    for line in src:
        for old, new in replace_map.items():
            line = line.replace(old, new)
        new_src.append(line)
    nb['cells'][idx]['source'] = new_src

Path('notebooks/mfg_pipeline.ipynb').write_text(json.dumps(nb, indent=2), encoding='utf-8')
