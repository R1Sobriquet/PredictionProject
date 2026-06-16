# 📊 Dictionnaire de Données - Fichier CSV Final Enrichi

**Fichier :** `data/processed/commandes_enriched.csv`  
**Source :** Pipeline d'ingestion et d'enrichissement  
**Période :** Janvier à Novembre 2024  
**Granularité :** 1 ligne par jour ET par article (même si quantité = 0)

---

## 🔹 **COLONNES DE BASE** (Issues de l'ingestion)

| Nom dans le CSV   | Description                                | Comment elle est obtenue                                                                                                                                                                                                                                                               |
|-------------------|--------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`date`**        | Date de la commande (YYYY-MM-DD)           | **Source :** Colonne `date_ligne_commande` du fichier brut<br>**Traitement :** Standardisée au format datetime<br>**Filtre :** Conserve uniquement 2024-01-01 à 2024-11-30<br>**Complétude :** Tous les jours de la période sont présents (même sans commande)                         |
| **`article_id`**  | Identifiant numérique de l'article         | **Source :** Colonne `id_article` du fichier brut<br>**Traitement :** Converti en entier<br>**Validation :** Doit être un nombre positif<br>**Unicité :** Combiné avec `date` forme une clé unique                                                                                     |
| **`quantity`**    | Quantité commandée (peut être 0)           | **Source :** Colonne `quantite` du fichier brut<br>**Traitement :** <br>- Agrégée (somme) si plusieurs lignes même jour/article<br>- Valeurs négatives supprimées<br>- Valeurs aberrantes (>10000) supprimées<br>- **0 ajouté** pour les jours sans commande<br>**Type :** Entier >= 0 |
| **`article_ref`** | Référence métier de l'article (ex: REF001) | **Source :** Colonne `ref_article` du fichier brut (optionnel)<br>**Traitement :** Conservée telle quelle<br>**Usage :** Pour identifier l'article de manière humaine                                                                                                                  |

---

## 🔹 **COLONNES TEMPORELLES** (Ajoutées par enrichissement)

| Nom dans le CSV    | Description                             | Comment elle est obtenue                                                                                                                                                                               |
|--------------------|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`year`**         | Année (2024)                            | **Calcul :** Extrait de la colonne `date` via `date.dt.year`<br>**Valeur :** Toujours 2024 dans ce dataset<br>**Usage :** Pour filtres temporels ou groupements                                        |
| **`month`**        | Mois (1 à 11)                           | **Calcul :** Extrait de `date` via `date.dt.month`<br>**Valeurs possibles :** 1 (janvier) à 11 (novembre)<br>**Usage :** Analyser la saisonnalité mensuelle                                            |
| **`day`**          | Jour du mois (1 à 31)                   | **Calcul :** Extrait de `date` via `date.dt.day`<br>**Valeurs possibles :** 1 à 31 selon le mois<br>**Usage :** Identifier les jours spécifiques (ex: début/fin de mois)                               |
| **`weekday`**      | Jour de la semaine (0 à 6)              | **Calcul :** Extrait de `date` via `date.dt.weekday`<br>**Mapping :** 0=Lundi, 1=Mardi, 2=Mercredi, 3=Jeudi, 4=Vendredi, 5=Samedi, 6=Dimanche<br>**Usage :** Capturer les patterns hebdomadaires       |
| **`weekday_name`** | Nom du jour (Lundi, Mardi...)           | **Calcul :** Mapping de `weekday` avec le dictionnaire `WEEKDAY_NAMES`<br>**Valeurs :** Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi, Dimanche<br>**Usage :** Affichage lisible dans les graphiques |
| **`is_weekend`**   | Indicateur weekend (True/False)         | **Calcul :** `True` si `weekday` est 5 (Samedi) ou 6 (Dimanche)<br>**Type :** Boolean<br>**Usage :** Analyser la différence weekend vs semaine                                                         |
| **`week_number`**  | Numéro de semaine dans l'année (1 à 52) | **Calcul :** Extrait de `date` via `date.dt.isocalendar().week`<br>**Standard :** ISO 8601 (semaine commence le lundi)<br>**Usage :** Agréger par semaine                                              |
| **`day_of_year`**  | Jour de l'année (1 à 365)               | **Calcul :** Extrait de `date` via `date.dt.dayofyear`<br>**Valeurs :** 1 (1er janvier) à 334 (30 novembre)<br>**Usage :** Capturer la saisonnalité annuelle                                           |
| **`quarter`**      | Trimestre (1 à 4)                       | **Calcul :** Extrait de `date` via `date.dt.quarter`<br>**Valeurs :** Q1 (jan-mar), Q2 (avr-jun), Q3 (jul-sep), Q4 (oct-nov ici)<br>**Usage :** Analyses trimestrielles                                |

---

## 🔹 **COLONNES DE RETARD (LAG FEATURES)** (Ajoutées par enrichissement)

| Nom dans le CSV         | Description                       | Comment elle est obtenue                                                                                                                                                                                                                                                      |
|-------------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`quantity_prev_day`** | Quantité commandée la veille      | **Calcul :** Pour chaque article, décalage de 1 jour de la colonne `quantity` via `shift(1)`<br>**Première ligne :** 0 (pas de jour précédent)<br>**Groupement :** Par `article_id` (chaque article a son propre historique)<br>**Usage :** Capturer la dépendance temporelle |
| **`quantity_lag_1`**    | Alias de `quantity_prev_day`      | **Calcul :** Identique à `quantity_prev_day`<br>**Usage :** Standardisation des noms pour les modèles ML                                                                                                                                                                      |
| **`quantity_lag_7`**    | Quantité commandée il y a 7 jours | **Calcul :** Décalage de 7 jours via `shift(7)` groupé par article<br>**Premières lignes :** 0 (moins de 7 jours d'historique)<br>**Usage :** Capturer la saisonnalité hebdomadaire                                                                                           |

---

## 🔹 **COLONNES DE MOYENNES MOBILES** (Ajoutées par enrichissement)

| Nom dans le CSV                 | Description                   | Comment elle est obtenue                                                                                                                                                                                                                                                     |
|---------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`quantity_rolling_mean_7d`**  | Moyenne des 7 derniers jours  | **Calcul :** Moyenne mobile sur 7 jours via `rolling(7).mean()` groupée par article<br>**Type :** Nombre décimal (float) arrondi à 2 décimales<br>**Premières lignes :** Moyenne sur les jours disponibles (min_periods=1)<br>**Usage :** Lisser les variations quotidiennes |
| **`quantity_rolling_mean_30d`** | Moyenne des 30 derniers jours | **Calcul :** Moyenne mobile sur 30 jours via `rolling(30).mean()` groupée par article<br>**Type :** Nombre décimal arrondi à 2 décimales<br>**Usage :** Identifier les tendances à moyen terme                                                                               |

---

## 🔹 **COLONNES SAISONNIÈRES AVANCÉES** (Ajoutées par enrichissement)

| Nom dans le CSV       | Description                                    | Comment elle est obtenue                                                                                                                                    |
|-----------------------|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`is_month_start`**  | Début de mois (jours 1-5)                      | **Calcul :** `True` si `day` <= 5<br>**Usage :** Capturer les patterns de début de mois                                                                     |
| **`is_month_middle`** | Milieu de mois (jours 11-20)                   | **Calcul :** `True` si 10 < `day` <= 20<br>**Usage :** Capturer les patterns de milieu de mois                                                              |
| **`is_month_end`**    | Fin de mois (jours 26+)                        | **Calcul :** `True` si `day` > 25<br>**Usage :** Capturer les patterns de fin de mois                                                                       |
| **`day_of_year_sin`** | Composante sinusoïdale du jour de l'année      | **Calcul :** `sin(2π × day_of_year / 365.25)`<br>**Valeurs :** Entre -1 et 1<br>**Usage :** Capturer la saisonnalité annuelle de manière cyclique (pour ML) |
| **`day_of_year_cos`** | Composante cosinusoïdale du jour de l'année    | **Calcul :** `cos(2π × day_of_year / 365.25)`<br>**Valeurs :** Entre -1 et 1<br>**Usage :** Complète `day_of_year_sin` pour représenter le cycle annuel     |
| **`weekday_sin`**     | Composante sinusoïdale du jour de la semaine   | **Calcul :** `sin(2π × weekday / 7)`<br>**Valeurs :** Entre -1 et 1<br>**Usage :** Capturer la saisonnalité hebdomadaire de manière cyclique                |
| **`weekday_cos`**     | Composante cosinusoïdale du jour de la semaine | **Calcul :** `cos(2π × weekday / 7)`<br>**Valeurs :** Entre -1 et 1<br>**Usage :** Complète `weekday_sin` pour représenter le cycle hebdomadaire            |

---

## 📊 **STATISTIQUES DU DATASET FINAL**

| Métrique                     | Valeur Typique                               |
|------------------------------|----------------------------------------------|
| **Nombre total de lignes**   | 335 jours × N articles                       |
| **Période couverte**         | 2024-01-01 à 2024-11-30 (335 jours)          |
| **Nombre de colonnes**       | ~25 colonnes (base + enrichies)              |
| **Lignes avec quantity = 0** | Variable (jours sans commande)               |
| **Lignes avec quantity > 0** | Variable (jours avec commande)               |
| **Clé unique**               | `date` + `article_id`                        |
| **Valeurs manquantes**       | 0 (aucune, toutes les combinaisons existent) |

---

## ✅ **RÈGLES DE VALIDATION**

### **Intégrité des données**
- ✅ Pas de valeurs NULL dans les colonnes critiques (date, article_id, quantity)
- ✅ Pas de doublons sur la combinaison (date, article_id)
- ✅ `quantity` >= 0 et <= 10000
- ✅ Toutes les dates entre 2024-01-01 et 2024-11-30 présentes
- ✅ Chaque article apparaît 335 fois (une fois par jour)

### **Cohérence temporelle**
- ✅ Les dates sont continues (pas de trou)
- ✅ `quantity_prev_day` correspond à la quantité du jour précédent
- ✅ Les moyennes mobiles sont cohérentes avec les valeurs brutes

### **Types de données**
- ✅ `date` : datetime64
- ✅ `article_id`, `quantity` : int64
- ✅ `is_weekend`, `is_month_start`, etc. : bool
- ✅ Moyennes mobiles, sin/cos : float64

---

## 🎯 **EXEMPLE DE LIGNE**

```csv
date,article_id,quantity,article_ref,year,month,day,weekday,weekday_name,is_weekend,quantity_prev_day,quantity_rolling_mean_7d,is_month_start,day_of_year_sin,weekday_sin
2024-01-08,1,70,REF001,2024,1,8,0,Lundi,False,55,60.14,False,0.0434,-0.7818
```

**Lecture :**
- 70 unités de l'article 1 commandées le lundi 8 janvier 2024
- La veille (7 janvier) : 55 unités
- Moyenne sur 7 jours : 60.14 unités
- Début d'année (jour 8 de l'année)
- Lundi (début de semaine de travail)

---

## 📖 **GLOSSAIRE**

| Terme                 | Définition                                                                                      |
|-----------------------|-------------------------------------------------------------------------------------------------|
| **Agrégation**        | Somme des quantités si plusieurs lignes le même jour pour le même article                       |
| **Lag feature**       | Variable qui décale une valeur dans le temps (ex: valeur d'hier)                                |
| **Moyenne mobile**    | Moyenne calculée sur une fenêtre glissante de N jours                                           |
| **Variable cyclique** | Transformation sin/cos pour représenter des cycles (évite le problème du "dimanche=6, lundi=0") |
| **Granularité**       | Plus petit niveau de détail = 1 ligne par jour ET par article                                   |

---

**Date de dernière mise à jour :** 2026-04-16  
**Version du pipeline :** 1.1.0 (ajout moteur de commande + DMQ par coupure)

---

## 🏦 **COLONNES PAR COUPURE** (Moteur de commande — v1.1.0)

Ces colonnes correspondent aux billets **5 / 10 / 20 / 50 / 100 €** de la
documentation PredikATM. Elles alimentent le moteur de commande
(`src/commande/`) et sont produites par `CatBoostDmqForecaster` +
`CommandPipeline`.

### 🔸 **Soldes par coupure** (source — table HFSQL)

| Nom dans le CSV | Description | Comment elle est obtenue |
|-----------------|-------------|--------------------------|
| `soldes_5` / `_10` / `_20` / `_50` / `_100` | Nombre de billets présents par coupure | **Source :** `DC_SoldesDuJour_{c}` depuis HFSQL<br>**Type :** float<br>**Usage :** alimente la détection K7 HS et la simulation de solde |
| `k7hs_5` / `_10` / `_20` / `_50` / `_100` | Flag cassette hors service | **Source :** `DC_K7HS_{c}` depuis HFSQL (valeur brute)<br>**Recalculé** par `detect_k7hs()` sur historique (15 j / 3 j stale)<br>**Type :** bool |
| `ajustement_5..100` | Ajustements par coupure | **Source :** `DC_Ajustement_{c}` depuis HFSQL<br>**Usage :** traçabilité des corrections manuelles |

### 🔸 **DMQ par coupure** (calculées par enrichissement — v1.1.0)

Consommation Quotidienne Moyenne projetée pour chaque coupure.

| Nom dans le CSV | Description | Comment elle est obtenue |
|-----------------|-------------|--------------------------|
| `dmq_5` / `_10` / `_20` / `_50` / `_100` | DMQ prédit par coupure (billets/jour) | **Calcul :** moyenne glissante 28 j des baisses de solde par coupure<br>**Alternative :** prédiction ML via `CatBoostDmqForecaster.predict()`<br>**Source config :** `DMQ_SOURCE=ml\|historical` (.env)<br>**Usage :** entrée du `CommandPipeline` (étapes 1 & 4) |
| `dmq_volatilite` | Écart-type glissant 28 j du DMQ | **Calcul :** `groupby(atm_id).rolling(28).std()`<br>**Usage :** feature ML, anticipation variance |
| `dmq_trend_7j` | Pente linéaire du DMQ sur 7 derniers jours | **Calcul :** `np.polyfit(range(7), dmq_7j, deg=1)[0]`<br>**Usage :** détecter tendances courtes |
| `dmq_trend_28j` | Pente linéaire du DMQ sur 28 jours | **Calcul :** idem sur 28 j<br>**Usage :** tendance moyen terme |
| `dmq_debut_mois_ratio` | `dmq_jour / dmq_moyen` | **Calcul :** ratio DMQ courant / moyenne ATM<br>**Usage :** caler l'effet « DMQ de début de mois » |
| `volatilite_dmq` | Volatilité brute (source HFSQL) | **Source :** `DC_VolatiliteDmq` |

### 🔸 **Configuration automate** (une ligne par ATM)

| Nom dans le CSV | Description | Comment elle est obtenue |
|-----------------|-------------|--------------------------|
| `nb_cassettes_5` / `_10` / `_20` / `_50` / `_100` | Nombre de cassettes par coupure | **Source :** configuration ATM (HFSQL, à intégrer)<br>**Usage :** formule étape 2 `SEUIL_MAX × nb_cassettes` |
| `nb_conteneurs` | Nombre de conteneurs Axytrans | **Usage :** cap étape 3 (2 600 billets × nb_conteneurs) |
| `insurance_amount` | Assurance agence (€) | **Usage :** cap étape 6 (assurance agence) |
| `mode_livraison` | `"axytrans"` ou autre | **Usage :** déclenche les caps Axytrans (étape 3) |
| `mode_chargement` | `"clic-clac"` ou `"complement"` | **Usage :** sélection finale étape 5 (remplissage max vs complément) |
| `soldes_ratio_assurance` | Ratio `Σ soldes × coupure / insurance_amount` | **Calcul :** `add_dmq_features()`<br>**Usage :** indicateur de remplissage |

### 🔸 **Sortie du moteur de commande** (`commandes_predictives.csv`)

Colonnes produites par `CommandPipeline.run()` et écrites dans
`data/output/commandes_predictives.csv`.

| Nom dans le CSV | Type | Description | Règle de calcul |
|-----------------|------|-------------|-----------------|
| `atm_id` | int | Identifiant de l'automate | Clé primaire |
| `predictif_5` / `_10` / `_20` / `_50` / `_100` | int | Nombre de billets à commander par coupure | Étapes 2-6 du moteur (cf. `call_stack.md`) |
| `k7hs_5..100` | bool | Cassette hors service ? | Étape 0 : solde inchangé ≥ 3 j sur 15 j observés |
| **`is_command`** ⭐ | bool | **Commande existe-t-elle ?** | **`any(predictif_c > 0 for c in [5,10,20,50,100])`** |
| `is_command_exceptionnelle` | bool | Commande exceptionnelle déclenchée | Étape 4 : solde soir < DMQ sur ≥ 1 coupure non-HS |
| `alerte_risque_vide` | bool | Risque que l'ATM soit vide | Identique à la condition précédente (pré-autorisation) |
| `alerte_commande_supprimee` | bool | Commande annulée (< seuil min) | Étape 3 ou 6 : montant < `CMD_MIN_AMOUNT` (2 000 €) |
| `alerte_commande_precedente_non_chargee` | bool | Une `PendingCommand` a été fournie | Saisie via paramètre de `pipeline.run()` |
| `montant_total` | int | Montant total de la commande (€) | `Σ predictif_c × c` |

**⭐ Règle métier clé :** `is_command` = True si et seulement si **au moins
une coupure** a un `predictif_c > 0`. Cette règle provient directement de la
documentation PredikATM et est testée dans
`tests/test_commande.py::TestCommandCalculator::test_is_command_rule_any_positive`.

---

## 🧪 **COMMANDES CLI QUI PRODUISENT CES COLONNES**

| Étape | Commande | Colonnes produites |
|-------|----------|--------------------|
| Enrichissement | `python main.py --step enrichment` | Temporelles, lag, rolling, saisonnières, **DMQ features** (v1.1.0) |
| CatBoost | `python main.py --step catboost` | Prédictions `amount` (montant agrégé) |
| **Moteur commande** | `python main.py --step command --day-commande YYYY-MM-DD --day-livraison YYYY-MM-DD` | **`predictif_5..100`, `is_command`, alertes** |
| Benchmark | `python tests/benchmark_dmq.py` | `data/output/precision_comparison.csv` (MAE/RMSE/MAPE par coupure) |
