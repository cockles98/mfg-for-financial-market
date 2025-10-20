# Dados e preparação

## Fontes

- **COTAHIST B3 (2015–2025)**: arquivos diários de cotações (disponíveis no site da B3).  
  Não são redistribuídos neste repositório. Copie os arquivos originais para
  `data/raw/` e utilize os utilitários em `scripts/` para ingestão.

## Pipeline resumido

1. Normalização e limpeza (`scripts/ingest_cotahist*.py`) → gera
   `data/processed/cotahist_equities_extended.parquet`.
2. Construção da curva de oferta (quantis de volume/spread):

   ```bash
   python scripts/build_supply_curve.py \
     --input data/processed/cotahist_equities_extended.parquet \
     --output data/processed/supply_curve.csv
   ```

3. Atualização do solver com a curva e a sensibilidade desejada:

   ```bash
   python scripts/update_solver_config.py \
     --supply data/processed/supply_curve.csv \
     --config configs/baseline.yaml \
     --scale 5e-05 \
     --price-sensitivity 30.0
   ```

4. (Opcional) Execute `python -m mfg_finance.cli run --config configs/baseline.yaml --endogenous-price`
   para validar o ajuste (preço médio ≈ 0 com esses parâmetros).

> **Observação legal**: a B3 não permite redistribuir os arquivos COTAHIST.
> Garanta que possui permissão para usá-los antes de reproduzir o pipeline.
