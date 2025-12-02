# 📚 PILE DES APPELS - Pipeline de Prévision

Document technique détaillant chaque appel de fonction du pipeline, référencé par numérotation du diagramme.

---

## 🎯 Vue d'Ensemble

| Phase | Étapes | Fichier Principal | Méthodes Clés |
|-------|--------|-------------------|---------------|
| **Ingestion** | 0-6 | `src/data_ingestion.py` | `run_full_pipeline()` |
| **Enrichissement** | 7-12 | `src/data_processing.py` | `run_full_enrichment()` |
| **Visualisation** | 13-14 | `src/visualization.py` | `plot_*()` |
| **Modélisation** | 15-18 | `src/models/baseline.py` | `evaluate_all_baselines()` |
| **Prédiction** | 19-20 | `src/models/baseline.py` | `predict()` |

---

## 🚀 PHASE 0 : DÉMARRAGE ET CHOIX SOURCE

### Point d'Entrée

**Fichier:** `main.py`

```python
def run_ingestion_step():
    """Orchestre l'étape d'ingestion complète"""
```

**Configuration:**

**Fichier:** `src/utils/config.py`

```python
DATA_SOURCE = os.getenv('DATA_SOURCE', 'csv')  # 'csv' ou 'sqlserver'
```

---

## 📥 ÉTAPE 0A : Chargement CSV

**Condition:** `DATA_SOURCE == 'csv'`

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def _load_from_csv(self):
        """
        Charge les données depuis un fichier CSV local.
        
        Paramètres implicites:
            self.source_file_path: Path vers le CSV
        
        Returns:
            pd.DataFrame avec colonnes brutes
        
        Exceptions:
            FileNotFoundError: Si CSV n'existe pas
            ValueError: Si CSV vide ou mal formaté
        """
```

**Dépendances:**
- `pandas.read_csv()`
- `ColumnNames.SOURCE_DATE` depuis `config.py`

**Sortie Attendue:**
- DataFrame avec ~1000-20000 lignes
- Colonnes : `date_ligne_commande`, `id_article`, `quantite`, etc.

---

## 🗄️ ÉTAPE 0B : Chargement SQL Server

**Condition:** `DATA_SOURCE == 'sqlserver'`

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def _load_from_sqlserver(self):
        """
        Charge les données depuis SQL Server.
        
        Utilise:
            self.db_connector: Instance de SQLServerConnector
        
        Appelle:
            db_connector.fetch_commandes_data()
        
        Returns:
            pd.DataFrame avec données SQL Server
        
        Exceptions:
            ConnectionError: Si connexion SQL Server échoue
        """
```

### Pile d'Appels Détaillée

**Niveau 1:** `data_ingestion.py`
```python
def _load_from_sqlserver(self):
    """Charge depuis SQL Server"""
```

**Niveau 2:** `database_connector.py`
```python
def fetch_commandes_data(self, start_date=None, end_date=None, article_ids=None):
    """Récupère les données de commandes"""
```

**Niveau 3:** `database_connector.py`
```python
def execute_query(self, query, params=None):
    """Exécute une requête SQL"""
```

**Requête SQL Exécutée:**
```sql
SELECT 
    [date_ligne_commande],
    [id_article],
    [ref_article],
    [quantite]
FROM [dbo].[ligne_commande]
WHERE 1=1
ORDER BY [date_ligne_commande], [id_article]
```

**Sortie Attendue:**
- DataFrame avec ~20000 lignes
- Colonnes : `date_ligne_commande`, `id_article`, `ref_article`, `quantite`

---

## 💾 SNAPSHOT 1 : Données Brutes Chargées

**Fichier:** `src/data_ingestion.py`

```python
def save_intermediate_snapshot(data, stage_name):
    """
    Sauvegarde un snapshot intermédiaire.
    
    Args:
        data: DataFrame à sauvegarder
        stage_name: Nom du stage (ex: '01_raw_loaded')
    
    Fichier créé:
        data/snapshots/snapshot_01_raw_loaded.csv
    """
```

**Fichier Créé:**
- `data/snapshots/snapshot_01_raw_loaded.csv`
- Contenu : Données brutes depuis CSV ou SQL Server
- Taille : 1000-20000 lignes selon source

---

## 📝 ÉTAPE 1 : Standardisation des Colonnes

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def standardize_columns(self):
        """
        Renomme les colonnes selon notre convention.
        
        Mapping:
            date_ligne_commande  → date
            id_article           → article_id
            quantite            → quantity
            ref_article         → article_ref
        
        Validation:
            - Vérifie présence colonnes requises
            - Lève ValueError si colonnes manquantes
        
        Modifie:
            self.raw_data (in-place)
        """
```

**Dépendances:**
- `pandas.DataFrame.rename()`
- Configuration depuis `src/utils/config.py`

---

## 💾 SNAPSHOT 2 : Colonnes Standardisées

**Fichier Créé:**
- `data/snapshots/snapshot_02_standardized.csv`
- Colonnes : `date`, `article_id`, `quantity`, `article_ref`

---

## 📅 ÉTAPE 2 : Filtrage de la Période

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def filter_training_period(self):
        """
        Filtre sur 2024-01-01 à 2024-11-30.
        
        Étapes:
            1. Normaliser les dates (dt.normalize)
            2. Convertir dates training en Timestamp
            3. Appliquer mask de filtrage
        
        Résout:
            Bug "can't compare datetime.datetime to datetime.date"
        
        Modifie:
            self.raw_data (filtrée)
        """
```

### Pile d'Appels Détaillée

```python
# Appels successifs
def filter_training_period(self):
    """Filtre la période d'entraînement"""
    # 1. Normaliser les dates
    # 2. Convertir les bornes
    # 3. Créer le masque booléen
    # 4. Filtrer et copier
```

**Variables Utilisées:**
- `training_start = datetime(2024, 1, 1)`
- `training_end = datetime(2024, 11, 30)`

**Sortie Attendue:**
- ~18300 lignes (filtré de 20000)
- Période : 2024-01-01 à 2024-11-30 (335 jours)

---

## 💾 SNAPSHOT 3 : Données Filtrées

**Fichier Créé:**
- `data/snapshots/snapshot_03_filtered.csv`
- ~18300 lignes
- Période : 2024-01-01 à 2024-11-30

---

## 🧹 ÉTAPE 3 : Validation et Nettoyage

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def validate_and_clean_data(self):
        """
        Nettoie les données selon règles métier.
        
        Étapes:
            1. dropna sur colonnes critiques
            2. Filtre quantités invalides (< 0, > 10000)
            3. Conversion types (int)
            4. drop_duplicates
        
        Règles:
            MIN_QUANTITY = 0
            MAX_QUANTITY = 10000
        
        Modifie:
            self.raw_data (nettoyée)
        """
```

**Dépendances:**
- `pandas.DataFrame.dropna()`
- `pandas.DataFrame.drop_duplicates()`
- `numpy.isfinite()`
- `ValidationRules` depuis `config.py`

**Sortie Attendue:**
- ~18000 lignes (après nettoyage)
- ~250 doublons supprimés
- Quantités : toutes entre 0 et 10000

---

## 💾 SNAPSHOT 4 : Données Nettoyées

**Fichier Créé:**
- `data/snapshots/snapshot_04_cleaned.csv`
- ~18000 lignes
- Aucune valeur manquante
- Aucun doublon

---

## 📊 ÉTAPE 4 : Agrégation Quotidienne

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def aggregate_daily_data(self):
        """
        Agrège par jour et article (somme des quantités).
        
        Cas d'usage:
            Plusieurs lignes même jour/article → 1 ligne avec somme
        
        Opérations:
            1. groupby [date, article_id]
            2. sum de quantity
            3. reset_index
            4. sort_values [date, article_id]
        
        Sortie:
            self.clean_data
        """
```

### Pile d'Appels Détaillée

```python
def aggregate_daily_data(self):
    """Agrège les données par jour"""
    # 1. Grouper par date et article
    # 2. Sommer les quantités
    # 3. Réinitialiser l'index
    # 4. Trier
    # 5. Réinitialiser l'index final
```

**Exemple Transformation:**
```
AVANT agrégation:
date        article_id  quantity
2024-01-01  1          30
2024-01-01  1          20  ← même jour/article
2024-01-02  1          40

APRÈS agrégation:
date        article_id  quantity
2024-01-01  1          50  ← somme 30+20
2024-01-02  1          40
```

**Sortie Attendue:**
- ~6200 combinaisons uniques jour/article
- 335 jours × ~20 articles (mais pas toutes combinaisons présentes)

---

## 💾 SNAPSHOT 5 : Données Agrégées

**Fichier Créé:**
- `data/snapshots/snapshot_05_aggregated.csv`
- ~6200 lignes
- 1 ligne par combinaison jour/article existante

---

## 🔄 ÉTAPE 5 : Remplissage des Combinaisons

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def fill_missing_combinations(self):
        """
        Ajoute TOUTES les combinaisons jour/article avec qty=0 si manquant.
        
        Objectif:
            Avoir 1 ligne par jour ET par article (même si qty=0)
        
        Étapes:
            1. Générer toutes les dates (pd.date_range)
            2. Lister tous les articles (unique)
            3. Produit cartésien (MultiIndex.from_product)
            4. Merge LEFT avec données existantes
            5. fillna(0) pour quantités manquantes
        
        Sortie:
            self.final_data
        """
```

### Pile d'Appels Détaillée

```python
def fill_missing_combinations(self):
    """Remplit toutes les combinaisons manquantes"""
    # 1. Générer toutes les dates (335 jours)
    # 2. Obtenir tous les articles uniques (20 articles)
    # 3. Créer le produit cartésien (335 × 20 = 6700 combinaisons)
    # 4. Merge avec les données existantes
    # 5. Remplir les NaN avec 0
```

**Exemple Transformation:**
```
AVANT remplissage (clean_data):
date        article_id  quantity
2024-01-01  1          50
2024-01-02  1          40
# Manque: 2024-01-01 article 2

APRÈS remplissage (final_data):
date        article_id  quantity
2024-01-01  1          50
2024-01-01  2          0  ← AJOUTÉ avec qty=0
2024-01-02  1          40
2024-01-02  2          0  ← AJOUTÉ avec qty=0
```

**Sortie Attendue:**
- Exactement 6700 lignes (335 jours × 20 articles)
- ~444 lignes avec quantity=0 (jours sans commande)
- ~6256 lignes avec quantity>0

---

## 💾 SNAPSHOT 6 : Données Complétées

**Fichier Créé:**
- `data/snapshots/snapshot_06_filled.csv`
- Exactement 6700 lignes
- Toutes les combinaisons jour/article

---

## 💾 ÉTAPE 6 : Sauvegarde Finale Ingestion

### Appels de Fonction

**Fichier:** `src/data_ingestion.py`

```python
class DataIngestionPipeline:
    def save_clean_data(self, output_path=None):
        """
        Sauvegarde les données nettoyées finales.
        
        Par défaut:
            output_path = data/processed/commandes_clean.csv
        
        Format:
            CSV avec colonnes: date, article_id, quantity
            date_format='%Y-%m-%d'
            index=False
        """
```

**Fichier Créé:**
- `data/processed/commandes_clean.csv`
- 6700 lignes
- 3-4 colonnes : `date`, `article_id`, `quantity`, (optionnel: `article_ref`)

**Résumé Ingestion:**
```python
summary = {
    "data_source": "sqlserver",  # ou "csv"
    "total_lines": 6700,
    "unique_dates": 335,
    "unique_articles": 20,
    "total_quantity": 917744,
    "zero_quantity_lines": 444,
    "date_range": {
        "start": "2024-01-01",
        "end": "2024-11-30"
    }
}
```

---

## 📥 ÉTAPE 7 : Chargement pour Enrichissement

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def load_clean_data(self, file_path=None):
        """
        Charge les données nettoyées depuis CSV.
        
        Par défaut:
            file_path = data/processed/commandes_clean.csv
        
        parse_dates:
            [ColumnNames.DATE] avec format '%Y-%m-%d'
        
        Sortie:
            self.clean_data
        """
```

**Dépendances:**
- `pandas.read_csv()` avec `parse_dates`
- `get_file_path('clean')` depuis `config.py`

---

## 📅 ÉTAPE 8 : Ajout Variables Temporelles

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def add_temporal_features(self):
        """
        Ajoute 7 variables temporelles.
        
        Variables créées:
            - year : dt.year
            - month : dt.month
            - day : dt.day
            - weekday : dt.weekday (0=Lundi, 6=Dimanche)
            - weekday_name : map avec WEEKDAY_NAMES
            - is_weekend : weekday in [5,6]
            - week_number : dt.isocalendar().week
        
        Modifie:
            self.enriched_data (copie de clean_data)
        """
```

### Pile d'Appels Détaillée

```python
def add_temporal_features(self):
    """Ajoute les variables temporelles"""
    # Copie des données
    # Extraction des composantes temporelles
    # Mapping et conditions
```

**Exemple Résultat:**
```
date        article_id  quantity  year  month  day  weekday  weekday_name  is_weekend  week_number
2024-01-01  1          50        2024  1      1    0        Lundi         False       1
2024-01-06  1          30        2024  1      6    5        Samedi        True        1
```

**Sortie Attendue:**
- +7 colonnes temporelles
- Total colonnes : 3 (base) + 7 = 10 colonnes

---

## ⏮️ ÉTAPE 9 : Ajout Variables de Retard

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def add_lag_features(self, lag_days=None):
        """
        Ajoute les variables de retard (valeurs des jours précédents).
        
        Pour chaque lag:
            - groupby article_id
            - shift(lag) sur quantity
            - fillna(0) pour premières lignes
        
        Variables créées:
            - quantity_lag_1 (jour précédent)
            - quantity_lag_7 (semaine précédente)
            - quantity_prev_day (alias de lag_1)
        
        Modifie:
            self.enriched_data
        """
```

### Pile d'Appels Détaillée

```python
def add_lag_features(self, lag_days=None):
    """Ajoute les variables de retard"""
    # 1. Trier par article et date
    # 2. Pour chaque lag
    #    - Grouper par article
    #    - Décaler de lag positions
    #    - Remplir les NaN avec 0
    #    - Assigner
```

**Exemple Transformation:**
```
AVANT (après tri):
date        article_id  quantity
2024-01-01  1          50
2024-01-02  1          40
2024-01-03  1          60

APRÈS ajout lag_1:
date        article_id  quantity  quantity_lag_1  quantity_prev_day
2024-01-01  1          50        0               0  ← Pas de jour précédent
2024-01-02  1          40        50              50  ← qty du 01
2024-01-03  1          60        40              40  ← qty du 02
```

**Sortie Attendue:**
- +3 colonnes : `quantity_lag_1`, `quantity_lag_7`, `quantity_prev_day`
- Total colonnes : 10 + 3 = 13 colonnes

---

## 📈 ÉTAPE 10 : Ajout Moyennes Mobiles

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def add_rolling_features(self, windows=None):
        """
        Ajoute des moyennes mobiles.
        
        Pour chaque fenêtre:
            - groupby article_id
            - rolling(window).mean()
            - min_periods=1 (pour début série)
            - round(2)
        
        Variables créées:
            - quantity_rolling_mean_7d
            - quantity_rolling_mean_30d
        
        Modifie:
            self.enriched_data
        """
```

### Pile d'Appels Détaillée

```python
def add_rolling_features(self, windows=None):
    """Ajoute les moyennes mobiles"""
    # Pour chaque fenêtre
    #    - Nom de la colonne
    #    - Grouper par article
    #    - Appliquer rolling mean
    #    - Arrondir
```

**Exemple Calcul:**
```
Date        Quantity  rolling_mean_7d
2024-01-01  50        50.00  ← (50)/1 = 50
2024-01-02  40        45.00  ← (50+40)/2 = 45
2024-01-03  60        50.00  ← (50+40+60)/3 = 50
...
2024-01-08  70        51.43  ← (50+40+60+...)/7
```

**Sortie Attendue:**
- +2 colonnes : `quantity_rolling_mean_7d`, `quantity_rolling_mean_30d`
- Total colonnes : 13 + 2 = 15 colonnes

---

## 🌍 ÉTAPE 11 : Ajout Variables Saisonnières

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def add_seasonal_features(self):
        """
        Ajoute 8 variables saisonnières.
        
        Variables créées:
            - day_of_year : dt.dayofyear (1-365)
            - quarter : dt.quarter (1-4)
            - is_month_start : day <= 5
            - is_month_middle : 10 < day <= 20
            - is_month_end : day > 25
            - day_of_year_sin : sin(2π × day_of_year / 365.25)
            - day_of_year_cos : cos(2π × day_of_year / 365.25)
            - weekday_sin : sin(2π × weekday / 7)
            - weekday_cos : cos(2π × weekday / 7)
        
        Modifie:
            self.enriched_data
        """
```

### Pile d'Appels Détaillée

```python
def add_seasonal_features(self):
    """Ajoute les variables saisonnières"""
    # 1. Composantes temporelles
    # 2. Position dans le mois
    # 3. Variables cycliques
```

**Pourquoi sin/cos ?**
Variables cycliques pour ML : éviter discontinuité entre Dimanche(6) et Lundi(0).

**Sortie Attendue:**
- +8 colonnes saisonnières
- Total colonnes : 15 + 8 = 23-24 colonnes

---

## 💾 ÉTAPE 12 : Sauvegarde Finale Enrichissement

### Appels de Fonction

**Fichier:** `src/data_processing.py`

```python
class DataEnrichmentPipeline:
    def save_enriched_data(self, output_path=None):
        """
        Sauvegarde les données enrichies.
        
        Par défaut:
            output_path = data/processed/commandes_enriched.csv
        
        Format:
            CSV avec ~24 colonnes
            date_format='%Y-%m-%d'
            index=False
        """
```

**Fichier Créé:**
- `data/processed/commandes_enriched.csv`
- 6700 lignes
- ~24 colonnes (3 base + 7 temporelles + 3 lag + 2 rolling + 8 saisonnières)

**Colonnes Finales:**
- **Base (3):** `date`, `article_id`, `quantity`
- **Temporelles (7):** `year`, `month`, `day`, `weekday`, `weekday_name`, `is_weekend`, `week_number`
- **Lag (3):** `quantity_lag_1`, `quantity_lag_7`, `quantity_prev_day`
- **Rolling (2):** `quantity_rolling_mean_7d`, `quantity_rolling_mean_30d`
- **Saisonnières (8):** `day_of_year`, `quarter`, `is_month_start`, `is_month_middle`, `is_month_end`, `day_of_year_sin`, `day_of_year_cos`, `weekday_sin`, `weekday_cos`

---

## 📊 ÉTAPE 13-14 : Visualisations

### Appels de Fonction

**Fichier:** `src/visualization.py`

```python
class DataVisualization:
    def load_enriched_data(self, file_path=None):
        """Charge les données enrichies"""
    
    def plot_weekday_analysis(self, save_path=None):
        """Analyse par jour de semaine"""
    
    def plot_weekend_vs_weekday_comparison(self, save_path=None):
        """Comparaison weekend vs semaine"""
    
    def plot_daily_sales_by_article(self, article_id, save_path=None):
        """Ventes quotidiennes d'un article"""
    
    def show_lag_verification(self, n_rows=10):
        """Vérification des variables de retard"""
```

**Graphiques Générés:**
- `data/output/charts/weekday_analysis.png`
- `data/output/charts/weekend_comparison.png`
- `data/output/charts/article_{id}_analysis.png`

**Note:** Erreur Qt possible sur Windows, mais graphiques sauvegardés quand même.

---

## 🤖 ÉTAPE 15-18 : Modélisation

### Appels de Fonction

**Fichier:** `src/models/baseline.py`

| Métrique | Unité           | Sens                                                  |
| -------- | --------------- | ----------------------------------------------------- |
| MAE      | Unités réelles  | Erreur moyenne linéaire                               |
| RMSE     | Unités réelles  | Erreur moyenne pondérée par les grosses erreurs       |
| MAPE     | Pourcentage (%) | Erreur moyenne relative par rapport à la vraie valeur |

#### Création des modèles

```python
def create_baseline_suite():
    """Crée les 7 modèles de baseline"""
```

#### Split train/test

```python
def split_train_test(enriched_data):
    """Sépare les données en train et test (90/10)"""
```

#### Évaluation

```python
def evaluate_all_baselines(baselines, train_data, test_data):
    """Entraîne et évalue tous les modèles"""
```

**Fichier Créé:**
- `data/output/baseline_results.csv`
- Colonnes : `model`, `mae`, `rmse`, `mape`
- Trié par MAE croissant

---

## 🔮 ÉTAPE 19-20 : Prédictions

### Appels de Fonction

**Fichier:** `src/models/baseline.py`

```python
def generate_predictions(model, articles, start_date, end_date):
    """Génère les prédictions pour tous les articles sur une période donnée"""
```

**Fichier Créé:**
- `data/output/predictions_2025.csv`
- Colonnes : `date`, `article_id`, `prediction`
- 365+ jours × N articles

---

## 📋 Résumé des Fichiers Créés

| Étape | Fichier | Lignes | Colonnes |
|-------|---------|--------|----------|
| **Snapshots Intermédiaires** ||||
| 1 | `data/snapshots/snapshot_01_raw_loaded.csv` | ~20000 | 4 |
| 2 | `data/snapshots/snapshot_02_standardized.csv` | ~20000 | 4 |
| 3 | `data/snapshots/snapshot_03_filtered.csv` | ~18300 | 4 |
| 4 | `data/snapshots/snapshot_04_cleaned.csv` | ~18000 | 4 |
| 5 | `data/snapshots/snapshot_05_aggregated.csv` | ~6200 | 3 |
| 6 | `data/snapshots/snapshot_06_filled.csv` | 6700 | 3 |
| **Fichiers Finaux** ||||
| 6 | `data/processed/commandes_clean.csv` | 6700 | 3-4 |
| 12 | `data/processed/commandes_enriched.csv` | 6700 | ~24 |
| 14 | `data/output/charts/*.png` | - | - |
| 18 | `data/output/baseline_results.csv` | 7 | 4 |
| 20 | `data/output/predictions_2025.csv` | 7300+ | 3 |

---

## 🔧 Configuration pour Snapshots

### Activer/Désactiver les Snapshots

**Fichier:** `main.py`

```python
# Avec snapshots (par défaut)
data = pipeline.run_full_pipeline(save_snapshots=True)

# Sans snapshots (plus rapide)
data = pipeline.run_full_pipeline(save_snapshots=False)
```

### Variable d'Environnement

**Fichier:** `.env`
```env
SAVE_SNAPSHOTS=true  # ou false
```

---

## 📞 Support Technique

**Pour debug d'une étape spécifique:**

1. Consulter le snapshot correspondant
2. Vérifier les logs : `forecasting_pipeline.log`
3. Comparer avant/après transformation

**Exemple:**
```bash
# Problème à l'agrégation ?
diff data/snapshots/snapshot_04_cleaned.csv data/snapshots/snapshot_05_aggregated.csv
```

---

*Document maintenu à jour avec le code - Version 1.0.0*