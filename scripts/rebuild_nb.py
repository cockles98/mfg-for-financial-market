import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

old_cells = nb['cells']

new_cells = []
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# Mean Field Game – Data-driven Pipeline\n',
        '\n',
        'Este notebook calibra o modelo com dados da B3, executa o solver LQ e explica cada etapa de forma intuitiva.\n'
    ]
})
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## Narrativa geral\n',
        '- **Dados**: usamos COTAHIST limpo para extrair volatilidade, retornos, spreads e volume.\n',
        '- **Calibração**: heurísticas ajustam `nu`, `phi`, `gamma_T`, `eta0`, `eta1`.\n',
        '- **Solver**: executamos o Picard com salvaguardas.\n',
        '- **Relatórios**: resultados são salvos em `notebooks_output/`.\n'
    ]
})
new_cells.append(old_cells[2])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 1. Configuração de diretórios\n']
})
new_cells.append(old_cells[3])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 2. Calibração heurística\n']
})
new_cells.append(old_cells[4])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 3. Atualizar baseline\n']
})
new_cells.append(old_cells[5])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 4. Construção do grid e parâmetros\n']
})
new_cells.append(old_cells[6])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 5. Executar solver\n']
})
new_cells.append(old_cells[7])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 6. Clearing de preço (opcional)\n']
})
new_cells.append(old_cells[8])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 7. Salvar artefatos\n']
})
new_cells.append(old_cells[9])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 8. Visualizações\n']
})
new_cells.append(old_cells[10])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 9. Resumo final\n']
})
new_cells.append(old_cells[11])
new_cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '## 10. Interpretando os gráficos\n',
        '- `density.png`: inventário ao longo do tempo.\n',
        '- `value.png`: função valor HJB.\n',
        '- `alpha_cuts.png`: agressividade em diferentes tempos.\n',
        '- `convergence.png`: evolução do erro.\n',
        '- `price.png`: preço de clearing (se habilitado).\n'
    ]
})

nb['cells'] = new_cells
nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
