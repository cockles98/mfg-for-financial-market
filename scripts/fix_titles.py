import json
from pathlib import Path
nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())
nb['cells'][0]['source'] = ['# Mean Field Game – Data-driven Pipeline\n', '\n', 'Este notebook calibra o modelo com dados da B3, executa o solver LQ e explica cada etapa de forma intuitiva.\n']
nb['cells'][6]['source'] = ['## 2.1 Curva de oferta empírica\n', 'Quantis de volume e spread usados no clearing.\n']
nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
