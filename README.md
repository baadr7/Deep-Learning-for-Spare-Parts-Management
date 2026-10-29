# Deep Learning for Spare Parts Management

Projet de semestre 9, ENSAM. Pilotage conjoint de la maintenance et du stock de pièces de
rechange d'une flotte de 100 machines : un simulateur à événements discrets reproduit la flotte,
le magasin et l'équipe de techniciens ; des modèles d'apprentissage profond anticipent les pannes
et un agent d'apprentissage par renforcement décide des commandes et des maintenances
préventives. Le cahier des charges se trouve dans [`docs/cdc`](docs/cdc).

## Organisation du dépôt

| Dossier | Contenu |
| --- | --- |
| `configs/scenarios` | Scénarios de simulation (S1 nominal, S0 homogène) |
| `configs/policies` | Paramètres optimisés des politiques de référence |
| `src/spareparts/sim` | Simulateur SimPy : machines, stock, techniciens, qualité |
| `src/spareparts/policies` | Politiques P0, P1, P2 et oracle |
| `src/spareparts/api` | API FastAPI de l'application web |
| `src/spareparts/data` | Chargement du jeu de données Azure |
| `scripts` | Scripts d'optimisation et d'entraînement |
| `web` | Interface React (voir `web/DESIGN.md`) |
| `tests` | Tests du simulateur et de l'API |

## Installation

Prérequis : Python 3.11 avec [uv](https://docs.astral.sh/uv/), Node.js 20 ou plus récent.

```bash
uv sync
cd web && npm install
```

Les modules d'apprentissage (PyTorch, Stable-Baselines3, LightGBM) s'installent à part :

```bash
uv sync --extra ml
```

## Lancement

Dans un premier terminal, l'API :

```bash
uv run uvicorn spareparts.api.app:app --reload
```

Dans un second terminal, l'interface, puis ouvrir http://localhost:5173 :

```bash
cd web && npm run dev
```

## Données

Télécharger le jeu [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
et placer les cinq fichiers CSV dans `data/raw`. Le simulateur fonctionne sans eux ; ils servent
à la calibration et à l'entraînement des modèles de prédiction sur données réelles.

## Tests

```bash
uv run pytest
```

## Commandes utiles

| Commande | Effet |
| --- | --- |
| `uv run python scripts/optimize_p1.py --trials 40` | Règle la politique P1 et enregistre ses paramètres |
| `uv run python scripts/train_predictor.py` | Simule cinq années, entraîne LightGBM, GRU, TCN et LSTM, écrit `reports/metrics` |
| `uv run python scripts/train_rl.py --level n1` | Entraîne l'agent P4 (MaskablePPO, niveau N1) |
| `uv run python scripts/train_rl.py --level n2` | Entraîne l'agent P5 (niveau N2, décisions machine par machine) |
| `uv run python scripts/train_rl.py --level reduced --algo dqn` | Entraîne le DQN de comparaison |
| `uv run python scripts/train_bdq.py` | Entraîne le Branching Dueling Q-Network au niveau N2 |
| `uv run python scripts/choose_agent_mode.py` | Choisit, sur les graines de réglage, si P4 et P5 jouent leur action la plus probable ou tirent leurs actions |
| `uv run python scripts/tune_p3.py` | Règle les seuils de la politique prédictive P3 |
| `uv run python scripts/train_surrogate.py` | Construit le plan d'expériences et entraîne le modèle de substitution |
| `uv run python scripts/sensitivity.py` | Indices de Sobol calculés sur le modèle de substitution |
| `uv run python scripts/evaluate_policies.py --robustness` | Évaluation finale sur 30 années et trois situations dégradées |
| `uv run python scripts/export_report_data.py` | Prépare les données des graphiques et tableaux du rapport |
| `uv run tensorboard --logdir runs/rl` | Suit les courbes d'apprentissage des agents |
| `cd web && npm run e2e` | Teste le parcours principal dans Microsoft Edge (API et interface lancées) |
| `uv run ruff check src tests` | Vérifie le style du code Python |

Les politiques P4, P5 et BDQ apparaissent dans l'application dès que leur modèle existe dans
`models/rl`, et la page « Scénarios » répond instantanément une fois le modèle de substitution
entraîné.
