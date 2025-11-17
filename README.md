# 🚀 Projet de Prévision de Commandes

Un système professionnel de prévision de demande basé sur l'analyse de données historiques de commandes 2024 pour prédire les commandes 2025.

## 📋 Table des Matières

- [🎯 Objectif du Projet](#-objectif-du-projet)
- [🏗️ Architecture](#️-architecture-)
- [⚡ Installation Rapide](#-installation-rapide)
- [🚀 Utilisation](#-utilisation)
- [📊 Pipeline de Données](#-pipeline-de-données)
- [📈 Modèles Implémentés](#-modèles-implémentés)
- [📁 Structure du Projet](#-structure-du-projet)
- [🧪 Tests](#-tests)
- [📖 Documentation](#-documentation)

## 🎯 Objectif du Projet

Ce projet implémente un système complet de prévision de commandes avec :
- **Ingestion et nettoyage** des données 2024
- **Enrichissement** avec variables temporelles et saisonnières
- **Modèles de baseline** robustes pour établir des références
- **Visualisations** pour identifier les patterns cachés
- **Prédictions** pour décembre 2024 et 2025

### Cas d'usage
- Prévision de demande quotidienne par article
- Optimisation des stocks
- Planification de la production
- Analyse des patterns saisonniers

## 🏗️ Architecture

```
📦 forecasting_project/
├── 📂 src/                     # Code source principal
│   ├── 📄 data_ingestion.py    # Étape 1A: Ingestion et nettoyage
│   ├── 📄 data_processing.py   # Étape 1B: Enrichissement
│   ├── 📄 visualization.py     # Graphiques et analyses
│   ├── 📂 models/              # Modèles de prévision
│   │   ├── 📄 baseline.py      # 7 modèles de baseline
│   └── 📂 utils/               # Configuration et utilitaires
├── 📂 data/                    # Données du projet
│   ├── 📂 raw/                 # Données brutes (commandes_2024.csv)
│   ├── 📂 processed/           # Données nettoyées et enrichies
│   └── 📂 output/              # Résultats et graphiques
├── 📂 tests/                   # Tests unitaires
├── 📄 main.py                  # Script principal
├── 📄 requirements.txt         # Dépendances
└── 📄 README.md               # Cette documentation
```

## ⚡ Installation Rapide

### Prérequis
- Python 3.13.5 ou compatible
- Fichier CSV de données de commandes 2024

### Étapes d'installation

1. **Cloner le projet**
```bash
git clone <votre-repo>
cd forecasting_project
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Préparer les données**
- Placez votre fichier de données dans `data/raw/commandes_2024.csv`
- Format attendu : colonnes `date_ligne_commande`, `id_article`, `quantite`

4. **Tester l'installation**
```bash
python main.py --step ingestion
```

## 🚀 Utilisation

### Pipeline Complet (Recommandé)
```bash
# Exécute toutes les étapes automatiquement
python main.py --step all
```

### Étapes Individuelles

**Étape 1A : Ingestion et nettoyage**
```bash
python main.py --step ingestion
```
- Charge les données 2024 (sauf décembre)
- Nettoie les doublons et valeurs aberrantes  
- Crée 1 ligne par jour/article avec quantité (même 0)

**Étape 1B : Enrichissement**
```bash
python main.py --step enrichment
```
- Ajoute les variables temporelles (jour semaine, weekend, etc.)
- Calcule `quantité_jour_prec` et moyennes mobiles
- Identifie automatiquement le jour de semaine le plus fort

**Analyse et Visualisations**
```bash
python main.py --step analysis
```
- Génère les graphiques d'analyse des patterns
- Compare weekend vs semaine
- Sauvegarde dans `data/output/charts/`

**Modèles de Baseline**
```bash
python main.py --step baselines
```
- Entraîne 7 modèles de référence
- Évalue et classe les performances
- Sauvegarde les résultats dans `data/output/`

### Analyse d'Article Spécifique
```bash
# Analyse détaillée de l'article 123
python main.py --article 123
```

## 📊 Pipeline de Données

### 1️⃣ Ingestion (Étape 1A)

**Objectif** : Transformer les données brutes en dataset propre et complet

**Processus** :
- ✅ Charge le fichier CSV de commandes 2024
- ✅ Filtre sur la période janvier-novembre 2024
- ✅ Supprime les doublons et valeurs aberrantes
- ✅ Agrège par jour/article (somme si plusieurs lignes)
- ✅ **Ajoute les jours manquants avec quantité = 0**

**Résultat** : Dataset avec 1 ligne par jour ET par article, toujours avec une quantité

### 2️⃣ Enrichissement (Étape 1B)

**Objectif** : Ajouter des variables explicatives pour la modélisation

**Variables ajoutées** :
- 📅 **Temporelles** : jour semaine, mois, weekend, trimestre
- ⏮️ **Retard** : `quantité_jour_prec`, moyennes mobiles 7/30 jours
- 🔄 **Saisonnières** : variables cycliques sin/cos
- 📊 **Patterns** : début/milieu/fin de mois

**Livrables** :
- ✅ Graphique des ventes par article (identifie les motifs répétitifs)
- ✅ **Jour de semaine le plus fort** calculé automatiquement
- ✅ Vérification de `quantité_jour_prec` (affichage des 10 premières lignes)

### 3️⃣ Modélisation

**Objectif** : Établir des références solides avec des modèles simples

**7 Modèles de Baseline Implémentés** :
1. **Naïf** : Dernière valeur observée
2. **Moyenne Historique** : Moyenne de tout l'historique  
3. **Moyenne Mobile 7j** : Moyenne des 7 derniers jours
4. **Moyenne Mobile 30j** : Moyenne des 30 derniers jours
5. **Moyenne par Jour Semaine** : Capture la saisonnalité hebdomadaire
6. **Saisonnier Naïf** : Même jour semaine précédente
7. **Tendance Linéaire** : Extrapolation de la tendance récente

## 📈 Modèles Implémentés

### Architecture Modulaire

Tous les modèles héritent de `BaselineModel` avec interface standardisée :
```python
model = WeekdayMeanBaseline()
model.fit(train_data)
predictions = model.predict(article_id=123, prediction_dates=[date1, date2])
metrics = model.evaluate(test_data)
```

### Évaluation Automatique

- **Métriques** : MAE, RMSE, MAPE
- **Division automatique** : 90% train, 10% test
- **Classement** : Modèles triés par performance
- **Robustesse** : Gestion des articles non vus

### Ensemble de Modèles

`BaselineEnsemble` combine plusieurs modèles avec pondération :
```python
ensemble = BaselineEnsemble([naive, weekday_mean], weights=[0.3, 0.7])
```

## 📁 Structure du Projet

### Fichiers Principaux

| Fichier | Rôle | Status |
|---------|------|---------|
| `main.py` | 🚀 Script principal et CLI | ✅ |
| `src/data_ingestion.py` | 📥 Étape 1A du pipeline | ✅ |
| `src/data_processing.py` | ⚙️ Étape 1B du pipeline | ✅ |
| `src/visualization.py` | 📊 Graphiques et analyses | ✅ |
| `src/models/baseline.py` | 🎯 7 modèles de baseline | ✅ |
| `src/utils/config.py` | ⚙️ Configuration centralisée | ✅ |

### Données Générées

```
data/
├── raw/commandes_2024.csv          # Vos données source
├── processed/
│   ├── commandes_clean.csv         # Après étape 1A
│   └── commandes_enriched.csv      # Après étape 1B
└── output/
    ├── baseline_results.csv        # Performance des modèles
    ├── charts/                     # Graphiques d'analyse
    └── articles/                   # Analyses par article
```

## 🧪 Tests

### Tests Unitaires
```bash
# Lancer tous les tests
pytest tests/

# Tests spécifiques
pytest tests/test_data_ingestion.py
pytest tests/test_baseline.py
```

### Validation des Données
```bash
# Aperçu des données brutes
python -c "from src import preview_raw_data; preview_raw_data('data/raw/commandes_2024.csv')"

# Vérification des variables de retard
python -c "from src import DataVisualization; viz = DataVisualization(); viz.load_enriched_data(); viz.show_lag_verification()"
```

## 📖 Documentation

### Logs Détaillés

Tous les traitements génèrent des logs détaillés :
- 📄 `forecasting_pipeline.log` : Log complet
- 🖥️ Console : Messages principaux en temps réel

### Métriques de Qualité

**Étape 1A - Ingestion** :
- Nombre de lignes avant/après nettoyage
- Articles uniques identifiés
- Période couverte
- Lignes avec quantité = 0

**Étape 1B - Enrichissement** :
- Jour de semaine le plus fort (automatique)
- Ratio weekend vs semaine
- Validation des variables de retard

**Modèles - Baseline** :
- Classement par MAE/RMSE
- Performance par article
- Temps d'entraînement

### Résolution de Problèmes

**Erreur : Fichier non trouvé**
```bash
# Vérifier les chemins
python -c "from src.utils import get_file_path; print(get_file_path('raw'))"
```

**Erreur : Colonne manquante**
- Vérifiez le mapping dans `src/utils/config.py`
- Adaptez `ColumnNames.SOURCE_*` selon vos colonnes

**Performance faible des modèles**
- Vérifiez la qualité des données (pas trop de valeurs à 0)
- Testez d'autres modèles ou ajustez les paramètres
- Analysez les patterns avec les visualisations

## 🤝 Contribution

### Architecture Extensible

- **Nouveaux modèles** : Hériter de `BaselineModel`
- **Nouvelles métriques** : Ajouter dans la méthode `evaluate()`
- **Nouveaux graphiques** : Étendre `DataVisualization`

### Standards de Code

- **Formatage** : `black src/`
- **Linting** : `flake8 src/`
- **Tests** : Couverture minimale 80%
- **Documentation** : Docstrings pour toutes les fonctions

---

## 🎉 Prêt à Utiliser !

```bash
# Démarrage rapide
python main.py --step all

# Ou étape par étape pour comprendre
python main.py --step ingestion
python main.py --step enrichment
python main.py --step analysis
python main.py --step baselines
```

**🔥 Fonctionnalités Clés Livrées :**
- ✅ Pipeline complet d'ingestion et nettoyage
- ✅ Enrichissement avec variables temporelles
- ✅ **Identification automatique du jour le plus fort**
- ✅ **Vérification de `quantité_jour_prec`**
- ✅ 7 modèles de baseline robustes
- ✅ Système d'évaluation automatique
- ✅ Graphiques d'analyse des patterns
- ✅ Architecture professionnelle et extensible

---

*Projet conçu pour la prévision de commandes industrielle avec focus sur la robustesse et la reproductibilité.* ⚡