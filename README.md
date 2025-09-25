# PredictionBTC – V0.51

Multi-horizon (H=4/8/16) + filtre H1 + hystérésis + logger & évaluation corrélation.

## Installation
```
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Entraînement
```
python -m model.train_multi
```

## Lancement du bot
```
python main.py
```

## Évaluation (corrélation / calibration / timeseries)
```
python -m analysis.eval_corr
```
Résultats dans `logs/analysis/`.
