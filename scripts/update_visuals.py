import json
from pathlib import Path

nb_path = Path('notebooks/mfg_pipeline.ipynb')
nb = json.loads(nb_path.read_text())

# modify code cell 18 (index 18) to display images inline
cell18 = nb['cells'][18]
cell18['source'] = [
    "density_path = REPORT_DIR / 'density.png'\n",
    "value_path = REPORT_DIR / 'value.png'\n",
    "alpha_path = REPORT_DIR / 'alpha_cuts.png'\n",
    "conv_path = REPORT_DIR / 'convergence.png'\n",
    "price_path = REPORT_DIR / 'price.png'\n",
    "\n",
    "plot_density_time(M_all, grid, density_path)\n",
    "plot_value_time(U_all, grid, value_path)\n",
    "plot_alpha_cuts(alpha_all, grid, times=[0.0, 0.25 * grid.T, 0.5 * grid.T], path=alpha_path)\n",
    "plot_convergence(errors, conv_path)\n",
    "if price_results is not None:\n",
    "    plot_price(grid.t, price_results, price_path)\n",
    "\n",
    "print('Figuras salvas em', REPORT_DIR)\n",
    "display(Image(filename=density_path))\n",
    "display(Image(filename=value_path))\n",
    "display(Image(filename=alpha_path))\n",
    "display(Image(filename=conv_path))\n",
    "if price_results is not None:\n",
    "    display(Image(filename=price_path))\n",
]

# also adjust import cell to include IPython display Image
cell2 = nb['cells'][2]
if 'from IPython.display import display' in ''.join(cell2['source']) and 'Image' not in ''.join(cell2['source']):
    idx = cell2['source'].index('from IPython.display import display\n')
    cell2['source'][idx] = 'from IPython.display import display, Image\n'

nb_path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
