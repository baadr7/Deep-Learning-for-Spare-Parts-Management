# Deep Learning for Spare Parts Management

**Pilotage conjoint de la maintenance et du stock de pièces de rechange d'une flotte de machines**

Projet de semestre 9 — École Nationale Supérieure d'Arts et Métiers (ENSAM) Meknès
Auteurs : **MA Bilal** et **Adam Elmadani** — 2026

---

## Résumé

Ce projet étudie le pilotage conjoint de la maintenance et du stock de pièces de rechange d'une
flotte de 100 machines réparties en deux familles, l'une vieillissant à l'usage, l'autre au
calendrier. Un simulateur à événements discrets reproduit pannes, maintenances, approvisionnements,
défauts de fabrication et télémétrie ; calibré sur le jeu *Microsoft Azure Predictive Maintenance*,
il retrouve le nombre de pannes de chaque composant sur l'année 2015 à moins de 10 % près. Des
réseaux de survie récurrents et convolutifs estiment le risque de panne de chaque composant à
partir de 48 heures de signaux ; ils dépassent LightGBM sur données simulées et, à 7 jours, sur
données réelles. Une politique prédictive fondée sur ces réseaux réduit le coût annuel de 5,6 % par
rapport à une politique à seuils fixes optimisés — écart significatif sur 30 années simulées — mais
une maintenance conditionnelle simple fait mieux encore, avec 9,6 % d'économie. Les agents
d'apprentissage par renforcement profond (PPO avec masquage d'actions, DQN, Branching Dueling
Q-Network) n'atteignent pas la politique à seuils optimisés ; le meilleur, qui règle chaque jour les
seuils de maintenance et de stock, s'en approche à 1,3 % près, tandis que les agents qui décident
machine par machine restent loin derrière. Un modèle de substitution ($R^2$ de 0,99 sur le coût)
permet l'analyse de sensibilité de Sobol, et une application web restitue l'ensemble.

## Sommaire

- [Deep Learning for Spare Parts Management](#deep-learning-for-spare-parts-management)
  - [Résumé](#résumé)
  - [Sommaire](#sommaire)
  - [1. Contexte et objectifs](#1-contexte-et-objectifs)
  - [2. Démarche](#2-démarche)
  - [3. Organisation du dépôt](#3-organisation-du-dépôt)
  - [4. Installation](#4-installation)
  - [5. Données](#5-données)
  - [6. Lancement](#6-lancement)
  - [7. Application web](#7-application-web)
  - [8. Reproduire les résultats](#8-reproduire-les-résultats)
    - [Référence des scripts](#référence-des-scripts)
  - [9. Tests et intégration continue](#9-tests-et-intégration-continue)
  - [10. Suivi des expériences](#10-suivi-des-expériences)
  - [11. Graines aléatoires](#11-graines-aléatoires)
  - [12. Résultats de référence](#12-résultats-de-référence)
  - [13. Rapport et documentation](#13-rapport-et-documentation)
  - [14. Citation et licence](#14-citation-et-licence)

---

## 1. Contexte et objectifs

Le projet répond à la question directrice suivante : **dans quelle mesure des modèles
d'apprentissage profond améliorent-ils conjointement la disponibilité d'une flotte et le coût de
gestion de ses pièces de rechange, par rapport à des règles de gestion classiques ?**

Il s'organise autour de quatre contributions, chacune mesurée sur un protocole commun :

1. **un simulateur à événements discrets** de la flotte, du magasin, de l'équipe technique et de la
   qualité, calibré sur des données industrielles réelles ;
2. **des modèles de prédiction de panne** (réseaux de survie GRU, TCN, LSTM ; référence LightGBM) ;
3. **des politiques de décision** allant de règles de seuils fixes à des agents d'apprentissage par
   renforcement profond, comparées entre elles et à un oracle omniscient ;
4. **une application web** industrielle qui restitue les simulations, les comparaisons, l'évaluation
   statistique et l'analyse de sensibilité.

## 2. Démarche

Le projet suit une chaîne expérimentale unique, reproductible de bout en bout (chapitre
« Reproduire les résultats » du rapport) :

```
Données Azure ──▶ calibration des lois de durée de vie ──▶ simulateur à événements discrets
                                                                     │
                           ┌─────────────────────────────────────────┼──────────────────────────┐
                           ▼                                         ▼                          ▼
              modèles de prédiction (GRU/TCN/LSTM)      politiques (P0…P5, DQN, BDQ)      plan d'expériences
                           │                                         │                          │
                           └──────────────▶ évaluation comparative ◀┘                          │
                                          (Wilcoxon, TOST ±2 %, IC)                             │
                                                     │                                          ▼
                                                     │                            modèle de substitution + Sobol
                                                     ▼
                                            application web (FastAPI + React)
```

Chaque politique est jugée sur les mêmes années que ses concurrentes (graines partagées), et les
graines d'évaluation sont postérieures à celles du réglage, afin d'écarter tout biais optimiste
(voir [§ 11](#11-graines-aléatoires)).

## 3. Organisation du dépôt

| Chemin | Contenu |
| --- | --- |
| `configs/scenarios` | Scénarios de simulation (S1 hétérogène nominal, S0 homogène) |
| `configs/policies` | Paramètres optimisés des politiques de référence (P1, P3, P4, P5) |
| `configs/fleets` | Compositions de flotte tirées pour l'étude de généralisation |
| `configs/calibration` | Lois de Weibull estimées sur le jeu Azure |
| `src/spareparts/sim` | Simulateur SimPy : machines, stock, techniciens, qualité, télémétrie |
| `src/spareparts/policies` | Politiques P0, P1, P2, P3, P4, P5, oracle |
| `src/spareparts/predict` | Modèles de prédiction de panne et jeux d'apprentissage |
| `src/spareparts/reliability` | Analyse de survie (Weibull) |
| `src/spareparts/rl` | Espaces d'action et réseaux d'apprentissage par renforcement |
| `src/spareparts/surrogate` | Plan d'expériences et modèle de substitution |
| `src/spareparts/api` | API FastAPI de l'application web |
| `src/spareparts/data` | Chargement et contrôle du jeu de données Azure |
| `models` | Modèles entraînés (`predict`, `rl`, `surrogate`) et résultats produits |
| `reports` | Rapports d'évaluation et métriques brutes (JSON, Markdown) |
| `scripts` | Optimisation, entraînement, évaluation et export vers le rapport |
| `web` | Interface React (voir [`web/DESIGN.md`](web/DESIGN.md)) |
| `tests` | Tests du simulateur, des politiques, de la prédiction, de l'API et du rapport |
| `docs/cdc` | Cahier des charges ([PDF](docs/cdc/cahier_des_charges.pdf)) |
| `docs/rapport` | Rapport LaTeX et ses données (`data`, `figures`) |

## 4. Installation

Prérequis : **Python 3.11** avec [uv](https://docs.astral.sh/uv/) et **Node.js 20** ou plus récent.

```bash
uv sync                       # environnement Python (numpy, simpy, fastapi, …)
cd web && npm install         # interface (React 19, Vite 6, Vitest)
```

Les modules d'apprentissage s'installent séparément, car ils sont lourds et facultatifs pour
l'exécution du simulateur :

```bash
uv sync --extra ml            # torch, stable-baselines3, lightgbm, lifelines, SALib, mlflow
```

## 5. Données

Le simulateur fonctionne **sans données externes**. Pour reproduire la calibration, télécharger le
jeu [Microsoft Azure Predictive Maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
et placer les cinq fichiers CSV dans `data/raw` :

```
PdM_telemetry.csv   PdM_failures.csv   PdM_machines.csv   PdM_errors.csv   PdM_maint.csv
```

Ce jeu sert à la calibration des lois de durée de vie et à l'entraînement des modèles de prédiction
sur données réelles ; il n'est pas versionné (voir `data/raw/.gitkeep`). Les fichiers dérivés du
simulateur sont écrits dans `data/processed`.

## 6. Lancement

Dans un premier terminal, l'API :

```bash
uv run uvicorn spareparts.api.app:app --reload
```

Dans un second terminal, l'interface, puis ouvrir <http://localhost:5173> :

```bash
cd web && npm run dev
```

L'ensemble se lance aussi avec Docker Compose (API sur le port 8000, interface sur le port 5173) :

```bash
docker compose up --build
```

## 7. Application web

L'interface (React 19 / Vite 6, thème clair, chiffres en premier) expose neuf pages :

| Page | Rôle |
| --- | --- |
| Tableau de bord | Indicateurs clés et évolution du coût |
| Flotte en direct | Simulation diffusée en temps réel (WebSocket), magasin et techniciens |
| Machines | Vue tabulaire de la flotte, filtrable |
| Fiche machine | États, signaux, taux de défaut, alarmes et événements d'une machine |
| Stock | Niveau de stock, commandes en cours et machines en attente de pièce |
| Comparaison | Comparaison de politiques sur plusieurs réplications, avec intervalles |
| Évaluation | Rapport comparatif, robustesse, test TOST, généralisation et plancher d'action |
| Scénarios | Exploration par modèle de substitution (paramètres de charge, délais, stock) |
| Données | Synthèse du jeu Azure, calibration et métriques de prédiction |
| Reproductibilité | Graines, protocole et commandes de reproduction |

L'API expose 16 routes HTTP et un WebSocket :

| Route | Méthode | Objet |
| --- | --- | --- |
| `/api/health` | GET | État du service |
| `/api/scenarios` | GET | Scénarios disponibles |
| `/api/policies` | GET | Politiques disponibles |
| `/api/runs` | POST | Lance une simulation, renvoie le run complet |
| `/api/runs/{id}` | GET | Résultat d'une simulation |
| `/api/runs/{id}/machines/{machine}` | GET | Détail d'une machine (états, signaux, événements) |
| `/api/runs/{id}/events` | GET | Journal d'événements (filtre `kind`) |
| `/api/compare` | POST | Compare des politiques sur plusieurs réplications |
| `/api/whatif` | POST | Rejoue une politique avec surcharges (techniciens, délais, taux de panne) |
| `/api/data/summary` | GET | Synthèse du jeu Azure et de la calibration |
| `/api/models/prediction` | GET | Métriques des modèles de prédiction |
| `/api/evaluation` | GET | Rapport comparatif et de robustesse |
| `/api/generalisation` | GET | Généralisation du classement sur compositions de flotte |
| `/api/plancher` | GET | Plancher de l'espace d'action des agents |
| `/api/surrogate` | GET | État et métriques du modèle de substitution |
| `/api/surrogate/predict` | POST | Prédiction instantanée des indicateurs par le métamodèle |
| `/api/ws/live` | WS | Diffusion d'une simulation en direct (`start`, `day`, `pause`, `resume`, `stop`) |

## 8. Reproduire les résultats

Les commandes ci-dessous s'enchaînent dans l'ordre de la chaîne expérimentale. Elles supposent
l'extra `ml` installé ([§ 4](#4-installation)).

```bash
# 1. Calibrer les lois de durée de vie sur le jeu Azure, puis valider le simulateur
uv run python scripts/calibrate_azure.py
uv run python scripts/validate_azure.py

# 2. Entraîner les modèles de prédiction et la politique prédictive
uv run python scripts/train_predictor.py
uv run python scripts/tune_p3.py

# 3. Régler les politiques de référence et entraîner les agents
uv run python scripts/optimize_p1.py --trials 40
uv run python scripts/train_rl.py --level n1
uv run python scripts/train_rl.py --level n2
uv run python scripts/train_rl.py --level reduced --algo dqn
uv run python scripts/train_bdq.py
uv run python scripts/choose_agent_mode.py

# 4. Construire le modèle de substitution et l'analyse de sensibilité
uv run python scripts/train_surrogate.py --replications 5
uv run python scripts/sensitivity.py

# 5. Évaluer, mesurer la généralisation et le plancher d'action
uv run python scripts/evaluate_policies.py --robustness
uv run python scripts/generate_fleets.py --n 5
uv run python scripts/generalize_ranking.py --replications 30
uv run python scripts/action_space_floor.py

# 6. Préparer les données du rapport et le compiler
uv run python scripts/export_report_data.py
latexmk -xelatex -cd docs/rapport/rapport.tex
```

### Référence des scripts

| Commande | Effet |
| --- | --- |
| `scripts/calibrate_azure.py` | Estime les lois de Weibull par composant à partir des pannes Azure |
| `scripts/validate_azure.py` | Compare les pannes simulées aux pannes observées (critère < 10 %) |
| `scripts/train_predictor.py` | Simule cinq années, entraîne LightGBM, GRU, TCN et LSTM, écrit `reports/metrics` |
| `scripts/tune_p3.py` | Règle les seuils de la politique prédictive P3 |
| `scripts/optimize_p1.py` | Règle la politique à seuils fixes P1 et enregistre ses paramètres |
| `scripts/train_rl.py --level {reduced,n1,n2}` | Entraîne P4 (N1), P5 (N2) ou le DQN de comparaison (`--resume` prolonge un apprentissage) |
| `scripts/train_bdq.py` | Entraîne le Branching Dueling Q-Network au niveau N2 |
| `scripts/choose_agent_mode.py` | Choisit entre action la plus probable et tirage pour P4 et P5 |
| `scripts/train_surrogate.py` | Construit le plan d'expériences et entraîne le modèle de substitution |
| `scripts/sensitivity.py` | Calcule les indices de Sobol sur le modèle de substitution |
| `scripts/evaluate_policies.py --robustness` | Évalue 30 années et quatre situations de robustesse |
| `scripts/generate_fleets.py` | Tire des compositions de flotte à nombre de machines constant |
| `scripts/generalize_ranking.py` | Rejoue l'évaluation sur les compositions et mesure la stabilité du classement |
| `scripts/action_space_floor.py` | Décompose l'écart à l'oracle en part d'apprentissage et part de représentation |
| `scripts/export_report_data.py` | Prépare les données des graphiques et tableaux du rapport |

Les politiques P4, P5 et BDQ apparaissent dans l'application dès que leur modèle existe dans
`models/rl` ; la page « Scénarios » répond instantanément une fois le modèle de substitution
entraîné ; la page « Évaluation » restitue les rapports produits par `evaluate_policies.py`,
`generalize_ranking.py` et `action_space_floor.py`.

## 9. Tests et intégration continue

```bash
uv run ruff check src tests scripts    # style et import (E, F, I, B, UP)
uv run pytest --cov=spareparts         # simulateur, politiques, prédiction, API, rapport
cd web && npm run build                # vérification des types et compilation de production
cd web && npm run test                 # composants et utilitaires (Vitest)
cd web && npm run e2e                  # parcours principal (Playwright, canal msedge)
```

Le seuil de couverture est fixé à **80 %** dans [`pyproject.toml`](pyproject.toml). La suite couvre
le simulateur, les politiques, la prédiction, l'API et la cohérence entre le rapport LaTeX et ses
données (`tests/test_report.py`).

L'intégration continue ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) déclenche quatre
tâches à chaque poussée sur `main` et à chaque demande de fusion :

| Tâche | Contenu |
| --- | --- |
| `python` | `uv sync --extra ml`, `ruff check`, `pytest --cov` |
| `web` | `npm ci`, `npm run build`, `npm run test` |
| `e2e` | Démarrage de l'API et de l'interface, puis `npx playwright test` (canal `msedge`) |
| `rapport` | Compilation XeLaTeX du rapport via l'image Docker `texlive/texlive` |

Le parcours de bout en bout s'appuie uniquement sur les données simulées ; la page « Scénarios »
interroge le modèle de substitution versionné (`models/surrogate`), qui exige l'extra `ml` côté API.

## 10. Suivi des expériences

Chaque entraînement journalise ses paramètres, ses métriques et ses fichiers de résultats avec
MLflow, en complément des courbes TensorBoard :

```bash
uv run tensorboard --logdir runs/rl                                    # courbes d'apprentissage des agents
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db               # journal des expériences
```

## 11. Graines aléatoires

Les graines ne sont pas choisies au hasard : chaque étape dispose de sa plage, et les plages restent
disjointes pour qu'aucune politique ne soit évaluée sur les années ayant servi à la régler.

| Usage | Graines |
| --- | --- |
| Calibration des lois de durée de vie | 0 à 3 |
| Jeu d'apprentissage de la prédiction (cinq années simulées) | 1 à 5 |
| Réglage de P1, de P3 et du mode de jeu des agents | 101, 102, 103 |
| Entraînement des agents (trois graines par agent) | 0, 1, 2 |
| Sélection du meilleur modèle des agents appris | 8000 à 8004 |
| Évaluation finale (30 réplications) et robustesse | 9000 à 9029 |

La même graine produit les mêmes pannes dans toutes les politiques : à l'intérieur d'une campagne,
les politiques affrontent donc les mêmes années, et l'écart de coût qui les sépare tient à la
politique et non au tirage. Les graines d'évaluation (9000 et suivantes) sont postérieures à toutes
celles du réglage et de la sélection, ce qui évite un biais optimiste.

## 12. Résultats de référence

Scénario nominal S1, 30 réplications (graines 9000-9029), horizon de 365 jours. Coûts en unités
relatives (UR) ; intervalles à 95 % ; le test TOST porte sur une marge de ±2 % du coût de P1.

| Politique | Coût total (UR) | TRS | Disponibilité | Ruptures |
| --- | --- | --- | --- | --- |
| Oracle (pannes futures connues) | 42408 [41755 ; 43061] | 97,29 % | 99,28 % | 10,2 % |
| P2 Maintenance conditionnelle | 47135 [45574 ; 48697] | 96,71 % | 98,85 % | 23,9 % |
| P3 Prédictive (sans apprentissage par renforcement) | 49220 [48252 ; 50189] | 96,83 % | 99,25 % | 2,1 % |
| P1 Seuils fixes optimisés *(référence)* | 52136 [50882 ; 53391] | 96,69 % | 99,06 % | 1,1 % |
| P4 Agent DRL, niveau N1 | 52799 [51232 ; 54365] | 96,44 % | 98,89 % | 3,9 % |
| DQN (niveau d'action réduit) | 54932 [53347 ; 56517] | 96,13 % | 98,45 % | 12,9 % |
| P0 Seuils fixes réglés à la main | 55355 [53702 ; 57008] | 96,08 % | 98,43 % | 12,4 % |
| P5 Agent DRL, niveau N2 | 67049 [65137 ; 68960] | 95,00 % | 97,97 % | 4,5 % |
| BDQ (niveau N2) | 74825 [71271 ; 78379] | 95,58 % | 98,45 % | 3,7 % |

Les classements s'entendent sous réserve de la variabilité inter-compositions mesurée en
généralisation. Le détail complet (tableaux de robustesse, écarts TOST, indices de Sobol,
généralisation, plancher d'action) se trouve dans [`reports/`](reports) et dans le rapport.

## 13. Rapport et documentation

| Document | Emplacement |
| --- | --- |
| Rapport complet (LaTeX) | [`docs/rapport/rapport.tex`](docs/rapport/rapport.tex) |
| Rapport compilé (PDF) | [`docs/rapport/rapport.pdf`](docs/rapport/rapport.pdf) |
| Données des figures et tableaux | `docs/rapport/data` |
| Cahier des charges | [`docs/cdc/cahier_des_charges.pdf`](docs/cdc/cahier_des_charges.pdf) |
| Système visuel de l'interface | [`web/DESIGN.md`](web/DESIGN.md) |

La compilation du rapport requiert XeLaTeX et `latexmk`, qui enchaîne les passes nécessaires
(sommaire, listes, références) :

```bash
latexmk -xelatex -cd docs/rapport/rapport.tex
```

Le rapport emploie les polices du poste de référence lorsqu'elles existent (Palatino Linotype,
Segoe UI, Cambria Math) et bascule sinon vers leurs équivalents libres (Pagella, Heros), de sorte
qu'il se compile indifféremment sous Windows ou sous TeX Live.

## 14. Citation et licence

Métadonnées de citation dans [`CITATION.cff`](CITATION.cff). Projet distribué sous licence MIT
(voir [`LICENSE`](LICENSE)).

```bibtex
@software{bilal_elmadani_spareparts_2026,
  title  = {Pilotage conjoint de la maintenance et du stock de pièces de rechange d'une flotte de machines},
  author = {Bilal, MA and Elmadani, Adam},
  year   = {2026},
  note   = {Projet de semestre 9, ENSAM Meknès}
}
```
