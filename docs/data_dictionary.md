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

**Date de dernière mise à jour :** 2025-10-15
**Version du pipeline :** 1.0.0  
