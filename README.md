# Projet de Prévision de Commandes ATM

Système de prévision de montants de rechargement ATM basé sur des données historiques exportées depuis HFSQL (WinDev).

**Objectif** : Etant donné qu'une commande de rechargement est déclenchée pour un ATM, prédire son montant (`DC_Montant_Cmd`).

## Architecture

```
PredictionProject/
├── main.py                      # Orchestration du pipeline
├── .env.example                 # Template de configuration
├── requirements.txt             # Dépendances Python
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py        # Chargement Excel (.xlsx) ou BDD
│   ├── data_processing.py       # Features ATM + temporelles
│   ├── database_connector.py    # MySQL + SQL Server
│   ├── visualization.py         # Graphiques et analyses
│   ├── models/
│   │   ├── baseline.py          # 7 modèles de référence
│   │   └── catboost_model.py    # Modèle CatBoost avancé
│   └── utils/
│       └── config.py            # Configuration centralisée
├── data/
│   ├── raw/                     # Fichiers Excel source
│   ├── processed/               # Données nettoyées/enrichies
│   └── output/                  # Résultats et modèles
└── tests/
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
DB_SERVER=10.147.18.196
DB_NAME=Python
DB_USER=sa
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
python main.py --step enrichment    # 2. Feature engineering ATM
python main.py --step analysis      # 3. Visualisations
python main.py --step baselines     # 4. Modèles de référence
python main.py --step catboost      # 5. Modèle CatBoost

# Analyse d'un ATM spécifique
python main.py --atm 123
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

## Tests

```bash
python -m pytest tests/ -v
```

## Résultats

Les résultats sont sauvegardés dans `data/output/` :
- `baseline_results.csv` : performances comparées des baselines
- `catboost_results_by_horizon.csv` : performances CatBoost par horizon
- `catboost_model/` : modèle sauvegardé
- `charts/` : graphiques d'analyse
