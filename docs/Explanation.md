# 🎯 Présentation du Projet de Prévision de Commandes

## 🌐 Vue d'ensemble

Système capable de **prédire les commandes 2025** en analysant l’historique **janvier → novembre 2024**.

---

## 📊 1. Récupération des données

### 🔄 Fonctionnement hybride

Le projet supporte deux sources configurables via `.env` :

```env
DATA_SOURCE=sqlserver   # ou "csv"
```

### **Mode SQL Server (recommandé)**

* Connexion directe via `pyodbc`
* Table : `dbo.ligne_commande`
* Requêtes SQL optimisées
* Module : `src/database_connector.py`

### **Mode CSV (legacy)**

* Chargement depuis `data/raw/commandes_2024.csv`
* Utile pour tests/démos sans BDD

**➡️ Avantage : un seul code, deux modes d'exécution selon l'environnement.**

---

## ⚙️ 2. Pipeline de traitement des données

### **Étape 1A — Ingestion (`data_ingestion.py`)**

Transforme les **données brutes → données propres**
Actions :

* Chargement SQL ou CSV
* Nettoyage : doublons, valeurs aberrantes, dates invalides
* Génération : **1 ligne / jour / article**, même si quantité = 0
* **Sortie :** `commandes_clean.csv`

---

### **Étape 1B — Enrichissement (`data_processing.py`)**

Transforme les **données propres → données enrichies** (25+ colonnes)

Ajouts :

* **Temporelles** : jour semaine, week-end, mois, trimestre
* **Retard (lag)** : quantité veille, moyennes mobiles 7 / 30 jours
* **Saisonnier** : transformées sin/cos hebdo + annuelles
* **Sortie :** `commandes_enriched.csv`

---

## 🎯 3. Modèles Baseline

### Qu'est-ce qu’une baseline ?

Modèles simples servant de **référence** pour comparer les futurs modèles ML.

> 🧠 *Si un modèle complexe ne bat pas la baseline, il ne sert à rien.*

### 🧩 Les 7 modèles implémentés

| Modèle                            | Principe                                | Exemple                 |
| --------------------------------- | --------------------------------------- | ----------------------- |
| Naïf                              | Répète la dernière valeur               | Hier : 50 → Demain : 50 |
| Moyenne historique                | Moyenne globale                         | Moyenne 2024 = 45       |
| Moyenne mobile (7/30j)            | Moyenne des N derniers jours            | 7 derniers jours = 48   |
| **Moyenne par jour de semaine ⭐** | Moyenne du même jour                    | Tous les lundis ≈ 60    |
| Saisonnier naïf                   | Valeur du même jour la semaine dernière | Lundi dernier = 55      |
| Tendance                          | Régression linéaire                     | Tendance +2/j → 52      |

**⭐ Recommandé : `WeekdayMeanBaseline` (capture très bien la saisonnalité).**

### Quand sont-ils utilisés ?

* **Entraînement :** Janvier → Novembre 2024
* **Évaluation :** Derniers 10 % des données
* **Prédiction :** Décembre 2024 + prévisions 2025
* **Comparaison :** Classement par MAE / RMSE

---

## 📈 4. Résultats & livrables

### Commande principale :

```bash
python main.py --step all
```

### Sorties générées :

```
data/output/
├── baseline_results.csv      # Performance et classement
├── predictions_2025.csv      # Prévisions 2025
└── charts/
    ├── weekday_analysis.png
    └── article_X_*.png
```

### Métriques clés

* **MAE** : erreur moyenne absolue
* **RMSE** : pénalise davantage les grosses erreurs

➡️ **Le modèle avec le MAE le plus bas est considéré comme le meilleur.**

---

## 💡 Points clés 

* **Flexibilité** : CSV ou SQL Server
* **Qualité** : pipeline robuste avec validation
* **Fiabilité** : 7 baselines solides
* **Extensible** : prêt pour intégrer du Machine Learning avancé
* **Production-ready** : logs, tests, documentation

---

## 🏦 5. Moteur de commande déterministe (module 4.1 PredikATM)

### 🎯 Règle métier clé

> **Si une des 5 valeurs par coupure (5 / 10 / 20 / 50 / 100 €) est > 0,
> alors c'est une commande.**

Autrement dit : `is_command = any(predictif_c > 0)`. Le moteur produit **5
valeurs par coupure** au lieu d'un simple montant agrégé, conformément à la
doc PredikATM.

### 🧩 Les 6 étapes (`src/commande/`)

| # | Module | Rôle | Règle clé |
|---|--------|------|-----------|
| **0** | `k7hs_detector.py` | Détection cassettes hors service | Solde inchangé ≥ 3 jours sur les 15 derniers → HS |
| **1** | `solde_simulator.py` | Projection au jour du chargement | `solde − 2,5 × DMQ` (hors férié) + commandes non chargées |
| **2** | `command_calculator.py` | Calcul nb billets par coupure | `max(0, SEUIL_MAX × nb_cassettes − solde)` |
| **3** | `verifications.py` | Vérifications min + Axytrans | Si total < 2 000 € → supprimée. Si Axytrans : cap 75 000 € / 2 600 billets × conteneur |
| **4** | `exceptional.py` | Commande exceptionnelle | Soir à `3,0 × DMQ` : si solde < DMQ → demi-seuil max |
| **5** | `pipeline.py` | Sélection finale | Exceptionnelle / clic-clac (seuil max) / complément (étape 2) |
| **6** | `insurance.py` | Caps assurance | Cap agence + cap global 300 000 € → revérifier min |

### 📤 Sortie du moteur

Fichier `data/output/commandes_predictives.csv`, une ligne par automate :

| Colonne | Description |
|---------|-------------|
| `predictif_5..100` | Nombre de billets à commander par coupure |
| `k7hs_5..100` | True si cassette HS (→ 0 billets commandés) |
| `is_command` | **any(predictif_c > 0)** — règle métier clé |
| `is_command_exceptionnelle` | True si étape 4 déclenchée |
| `alerte_risque_vide` | Solde soir < DMQ sur ≥ 1 coupure |
| `alerte_commande_supprimee` | Commande annulée (< seuil min) |
| `montant_total` | `Σ predictif_c × c` (€) |

### 🚀 Commande CLI

```bash
python main.py --step command \
    --day-commande 2026-03-30 \
    --day-livraison 2026-04-02
```

---

## 📈 6. Amélioration de la précision (prédiction DMQ par coupure)

### Avant / après (hyperparamètres CatBoost)

| Paramètre | Avant | Après | Gain attendu |
|-----------|-------|-------|--------------|
| `learning_rate` | **0.85** | **0.03** | Convergence stable (vs overfitting brutal) |
| `iterations` | 4444 | 4000 + early stop | Plus robuste |
| `l2_leaf_reg` | ∅ | **3.0** | Régularisation explicite |
| `subsample` / `rsm` | ∅ | **0.85 / 0.85** | Bagging Bernoulli |
| `loss_function` | `RMSE` | **`MAE`** | Robuste aux outliers |
| Target | Montant total | **5 × `dmq_{coupure}`** | Exploite la structure par coupure |

### Nouveau pipeline ML : `MultiCoupureForecaster`

Un modèle `CatBoostDmqForecaster` **par coupure** (5 modèles indépendants).
Exposé au moteur de commande via `as_dmq_provider()`.

### Validation temporelle

- `time_series_cv()` → 3-fold **growing window** (pas de fuite futur → passé)
- `evaluate_per_coupure()` → MAE / RMSE / MAPE par coupure + TOTAL
- Comparaison systématique vs `WeekdayMeanBaseline` (baseline recommandée)

### Features DMQ ajoutées (`add_dmq_features()`)

- `dmq_volatilite` (écart-type 28 j) — exploite `DC_VolatiliteDmq`
- `dmq_trend_7j` / `dmq_trend_28j` — pente linéaire régressée
- `dmq_debut_mois_ratio` — ratio DMQ du jour / DMQ moyen
- `soldes_ratio_assurance` — indicateur de remplissage

### 🎯 Critère de succès

> La MAE du `CatBoostDmqForecaster` par coupure doit battre celle du
> `WeekdayMeanBaseline` sur **≥ 4/5 coupures**.

Benchmark : `python tests/benchmark_dmq.py` → `data/output/precision_comparison.csv`.

---

## ✅ 7. Tests

- **`tests/test_commande.py`** — 28 tests unitaires (un par étape + 3 tests bout-en-bout)
- **`tests/benchmark_dmq.py`** — benchmark précision ML vs baseline

```bash
python -m pytest tests/test_commande.py -v   # 28 tests
python tests/benchmark_dmq.py                # Benchmark
```

---

## 🚀 Démonstration rapide

```bash
# 1. Configurer la source de données
nano .env   # DATA_SOURCE=sqlserver ou csv

# 2. Lancer le pipeline complet
python main.py --step all

# 3. Générer les commandes prédictives par coupure
python main.py --step command --day-commande 2026-03-30 --day-livraison 2026-04-02

# 4. Visualiser les résultats
cat data/output/baseline_results.csv
cat data/output/commandes_predictives.csv
```

Durée d'exécution : **1 à 3 minutes** selon la taille des données.

