# Projet de Prévision de Commandes ATM

Système de prévision de montants de rechargement ATM basé sur des données historiques exportées depuis HFSQL (WinDev).

**Objectifs** :
1. Prédire la consommation DMQ (Distribution Quotidienne Moyenne) **par coupure** (5 / 10 / 20 / 50 / 100 €).
2. Calculer la commande à passer via un **moteur déterministe en 6 étapes** (documentation PredikATM § 4.1).
3. **Règle métier clé** : une commande existe dès qu'une des 5 valeurs par coupure est supérieure à 0.

## Architecture

```
PredictionProject/
├── main.py                      # Orchestration du pipeline
├── .env.example                 # Template de configuration
├── requirements.txt             # Dépendances Python
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Chargement Excel (.xlsx) ou BDD
│   ├── data_processing.py       # Features ATM + temporelles + DMQ
│   ├── database_connector.py    # MySQL + SQL Server
│   ├── visualization.py         # Graphiques et analyses
│   ├── commande/                # Moteur de commande déterministe (6 étapes)
│   │   ├── k7hs_detector.py     # Étape 0 : cassettes hors service
│   │   ├── solde_simulator.py   # Étapes 1 & 4 : simulation des soldes
│   │   ├── command_calculator.py# Étape 2 : calcul par coupure
│   │   ├── verifications.py     # Étape 3 : min + caps Axytrans
│   │   ├── exceptional.py       # Étape 4 : commande exceptionnelle
│   │   ├── insurance.py         # Étape 6 : cap assurance + cap global
│   │   └── pipeline.py          # Orchestration des 6 étapes
│   ├── models/
│   │   ├── baseline.py          # 7 modèles de référence
│   │   ├── catboost_model.py    # CatBoost + MultiCoupureForecaster
│   │   └── evaluation.py        # MAE/RMSE/MAPE par coupure + CV temporelle
│   └── utils/
│       └── config.py            # Configuration centralisée + CommandConfig
├── data/
│   ├── raw/                     # Fichiers Excel source
│   ├── processed/               # Données nettoyées/enrichies
│   └── output/                  # Résultats et modèles
├── tests/
│   ├── test_commande.py         # 28 tests unitaires du moteur de commande
│   └── benchmark_dmq.py         # Benchmark précision ML vs baseline
├── examples/
│   └── catboost_example.py      # Exemple d'utilisation CatBoost
└── scripts/
    ├── generate_predictions.py  # Génère prédictions mensuelles/annuelles
    ├── inspect_excel.py         # Outil d'inspection des Excel source
    └── merge_commande_files.py  # Fusion de fichiers de commandes
```

## Stack Technique

- **Python 3.13**, pandas, numpy
- **CatBoost** (gradient boosting) + scikit-learn (baselines)
- **openpyxl** (lecture Excel .xlsx)
- **pyodbc** (SQL Server) + **pymysql** (MySQL)
- **python-dotenv** (configuration .env)

## Installation

```bash
git clone https://github.com/R1Sobriquet/PredictionProject.git
cd PredictionProject
pip install -r requirements.txt
cp .env.example .env
```

## Configuration (.env)

Le pipeline supporte deux modes de données, configurables via `.env` :

### Mode Excel (par défaut)

```env
DATA_SOURCE=csv
CSV_FILE_PATH=data/raw/d_CommandesDetailCalcul.xlsx
```

Place le fichier `d_CommandesDetailCalcul.xlsx` dans `data/raw/`.

### Mode Database (MySQL)

```env
DATA_SOURCE=database
DB_TYPE=mysql
DB_SERVER=localhost
DB_NAME=prediction_db
DB_USER=root
DB_PASSWORD=motdepasse
DB_TABLE=d_CommandesDetailCalcul
```

### Mode Database (SQL Server)

```env
DATA_SOURCE=database
DB_TYPE=sqlserver
DB_SERVER=x.x.x.x
DB_NAME=Python
DB_USER=user
DB_PASSWORD=motdepasse
DB_TABLE=d_CommandesDetailCalcul
DB_DRIVER=ODBC Driver 17 for SQL Server
```

## Utilisation

```bash
# Pipeline complet
python main.py --step all

# Étapes individuelles
python main.py --step ingestion     # 1. Chargement et nettoyage
python main.py --step enrichment    # 2. Feature engineering ATM + DMQ
python main.py --step analysis      # 3. Visualisations
python main.py --step baselines     # 4. Modèles de référence
python main.py --step catboost      # 5. Modèle CatBoost

# Moteur de commande déterministe (6 étapes) → predictif_5..100 + is_command
python main.py --step command \
    --day-commande 2026-03-30 \
    --day-livraison 2026-04-02

# Analyse d'un ATM spécifique
python main.py --atm 123
```

### Variables d'environnement (`.env`) — moteur de commande

Toutes les constantes métier sont surchargeables :

```env
# Seuils généraux
CMD_MIN_AMOUNT=2000                 # Seuil minimum d'une commande (€)
CMD_AXYTRANS_MAX_EUR=75000          # Cap montant Axytrans (€)
CMD_AXYTRANS_MAX_BILLETS_PER_CONTAINER=2600
CMD_INSURANCE_GLOBAL_CAP=300000     # Cap global agence (€)

# Seuils maximaux par coupure (billets par cassette)
CMD_SEUIL_MAX_5=2500
CMD_SEUIL_MAX_10=2500
CMD_SEUIL_MAX_20=2500
CMD_SEUIL_MAX_50=2500
CMD_SEUIL_MAX_100=2500

# Détection K7 HS (cassette hors service)
CMD_K7HS_WINDOW_DAYS=15             # Fenêtre d'observation
CMD_K7HS_STALE_DAYS=3               # Jours sans mouvement → HS

# Consommation DMQ pour la simulation
CMD_DMQ_CONSO_CHARGEMENT=2.5        # Jours de DMQ avant chargement
CMD_DMQ_CONSO_SOIR=3.0              # Jours de DMQ au soir

# Source du DMQ : "ml" (MultiCoupureForecaster) ou "historical" (moyenne 28j)
DMQ_SOURCE=historical
```

## Données

### Source

Fichier Excel `d_CommandesDetailCalcul.xlsx` exporté depuis une base HFSQL (WinDev).
Colonnes `DC_*` avec ~48 000 enregistrements sur 2026.

### Colonnes chargées (~25 sur 51)

| Colonne | Description |
|---------|-------------|
| `DC_Commande_Id` | ID unique de la commande |
| `DC_Automate_Id` | ID de l'ATM |
| `DC_Date_Cmd` | Date de la commande |
| `DC_Montant_Cmd` | **Target** : montant total (EUR) |
| `DC_Cassette_1..5` | Quantités par cassette |
| `DC_Ajustement_5/10/20/50/100` | Ajustements par coupure |
| `DC_SoldesDuJour_5/10/20/50/100` | Soldes du jour par coupure |
| `DC_K7HS_5/10/20/50/100` | Cassettes hors service |
| `DC_VolatiliteDmq` | Volatilité DMQ |
| `DC_Annule` | 1 = commande annulée (exclue) |
| `DC_RisqueAutomateVide` | Flag risque ATM vide |

**Exclus** : `DC_Predictif_*` (prédictions du système source = fuite de données).

### Features calculées

- **Historique ATM** : jours depuis dernier rechargement, montant précédent, fréquence moyenne, montant moyen/écart-type
- **Agrégats** : total soldes, total ajustements, total cassettes HS, cassettes actives
- **Délais** : délai livraison, délai chargement
- **Temporel** : weekday, weekend, mois, trimestre, encodage cyclique sin/cos
- **DMQ** (`add_dmq_features()`, *étape 5b*) :
  - `dmq_volatilite` : écart-type glissant 28 jours du DMQ
  - `dmq_trend_7j` / `dmq_trend_28j` : pente linéaire régressée sur N derniers jours
  - `dmq_debut_mois_ratio` : ratio DMQ du jour / DMQ moyen (cf. « DMQ de début de mois »)
  - `soldes_ratio_assurance` : indicateur de remplissage relatif à l'assurance agence
- **DMQ par coupure** (`add_dmq_per_coupure_features()`, *étape 5c*) :
  - `dmq_5`, `dmq_10`, `dmq_20`, `dmq_50`, `dmq_100` : moyenne glissante 28 j
    des baisses quotidiennes de `solde_<c>` (consommation observée, `shift(1)`
    pour éviter toute fuite). Utilisé comme signal d'entrée du moteur de
    commande et comme target pour `CatBoostDmqForecaster`.
- **Calendrier FR** (`src/utils/holidays.py`) :
  - `is_holiday` : jour férié français métropolitain (11 jours légaux)
  - `is_eve_holiday` : veille de férié (souvent un pic de consommation)
  - `is_payday` : fin de mois (≥28) ou 5 du mois (paie / pensions)

## Modèles

### Baselines (7 modèles de référence)

1. **Naive** : dernier montant observé
2. **Historical Mean** : montant moyen par ATM
3. **Moving Average (5/10 cmd)** : moyenne des N dernières commandes
4. **Weekday Mean** : montant moyen par jour de semaine
5. **Seasonal Naive** : même jour semaine précédente
6. **Trend** : tendance linéaire extrapolée

### CatBoost (modèle avancé)

- Horizon de prédiction 1-90 jours
- Features adaptatives court terme / long terme
- `atm_id` comme feature catégorielle native
- Early stopping pour éviter l'overfitting

**Hyperparamètres tunés** (cf. `catboost_model.py:67-118`) :

| Hyperparamètre   | Valeur | Justification                                      |
|------------------|--------|----------------------------------------------------|
| `learning_rate`  | 0.03   | Convergence stable (vs 0.85 → overfitting brutal)  |
| `iterations`     | 4000   | Plus d'itérations + early stopping                 |
| `depth`          | 6      | Profondeur standard                                |
| `l2_leaf_reg`    | 3.0    | Régularisation L2 sur les feuilles                 |
| `subsample`      | 0.85   | Bagging Bernoulli (+ `rsm=0.85`)                   |
| `loss_function`  | `MAE`  | Robuste aux outliers (queues épaisses des montants)|

### CatBoostDmqForecaster + MultiCoupureForecaster

Prédiction **par coupure** du DMQ : 5 modèles indépendants (un par coupure
5/10/20/50/100€). Cible = `dmq_{coupure}`. API de prédiction :

```python
from src.models import MultiCoupureForecaster

mcf = MultiCoupureForecaster()
mcf.fit(train_data)
dmq = mcf.predict_dmq_par_coupure(atm_id=123, prediction_date="2026-04-02")
# → {5: 102.3, 10: 78.5, 20: 55.1, 50: 40.9, 100: 21.4}

# Adaptateur vers le CommandPipeline
provider = mcf.as_dmq_provider(prediction_date="2026-04-02", context_data=train_data)
```

### Évaluation par coupure (`src/models/evaluation.py`)

- `evaluate_per_coupure(predictions, actuals)` → DataFrame `[coupure, mae, rmse, mape, n]` + ligne `TOTAL`
- `time_series_cv(model_factory, data, n_splits=3)` → CV temporelle (growing window, pas de fuite futur → passé)
- `compare_models_per_coupure({model_name: preds})` → tableau comparatif multi-modèles

## Moteur de commande déterministe (6 étapes)

Le pipeline `CommandPipeline` (`src/commande/pipeline.py`) applique
séquentiellement les règles métier de la doc PredikATM § 4.1 :

| Étape | Module                   | Rôle                                                    |
|-------|--------------------------|---------------------------------------------------------|
| 0     | `k7hs_detector.py`       | Détection cassettes HS (15j / 3j sans mouvement)        |
| 1     | `solde_simulator.py`     | Solde projeté au jour du chargement (2,5 × DMQ)         |
| 2     | `command_calculator.py`  | Formule `max(0, SEUIL_MAX × nb_cassettes − solde)`      |
| 3     | `verifications.py`       | Seuil min (2 000€) + caps Axytrans (75 000€ / 2 600 billets)|
| 4     | `exceptional.py`         | Soir (3,0 × DMQ) → commande exceptionnelle si risque vide|
| 5     | (pipeline)               | Sélection finale (exceptionnelle / clic-clac / complément)|
| 6     | `insurance.py`           | Cap assurance agence + cap global 300 000 €             |

**Sortie** (`data/output/commandes_predictives.csv`) : une ligne par automate
avec `predictif_5..100`, `k7hs_5..100`, `is_command`, `is_command_exceptionnelle`,
`alerte_*`, `montant_total`.

## Tests

```bash
# Tests unitaires du moteur de commande (28 tests, 6 étapes + bout-en-bout)
python -m pytest tests/test_commande.py -v

# Benchmark précision MultiCoupureForecaster vs WeekdayMeanBaseline
python tests/benchmark_dmq.py
# → data/output/precision_comparison.csv
```

**Critère de succès** : la MAE du `CatBoostDmqForecaster` par coupure doit
battre celle du `WeekdayMeanBaseline` sur ≥ 4/5 coupures.

## Validation croisée temporelle

Par défaut, baselines et CatBoost utilisent un split 90/10 temporel. Pour un
TimeSeriesSplit 3-fold expansif (growing window) :

```bash
python main.py --step baselines --cv timeseries    # -> data/output/baseline_results_cv.csv
python main.py --step catboost  --cv timeseries    # -> data/output/catboost_results_cv.csv

# Preset d'hyperparams CatBoost : fast | default | deep
python main.py --step catboost --catboost-preset fast
```

## Résultats

Les résultats sont sauvegardés dans `data/output/` :
- `baseline_results.csv` : performances comparées des baselines (split simple)
- `baseline_results_cv.csv` : performances baselines en TimeSeriesSplit (`--cv timeseries`)
- `catboost_results_by_horizon.csv` : performances CatBoost par horizon
- `catboost_results_cv.csv` : CatBoost en TimeSeriesSplit (`--cv timeseries`)
- `catboost_model/` : modèle sauvegardé
- `commandes_predictives.csv` : sortie du moteur de commande déterministe
- `precision_comparison.csv` : benchmark DMQ par coupure (ML vs baseline)
- `charts/` : graphiques d'analyse
