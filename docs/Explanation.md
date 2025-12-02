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

## 🚀 Démonstration rapide

```bash
# 1. Configurer la source de données
nano .env   # DATA_SOURCE=sqlserver ou csv

# 2. Lancer le pipeline complet
python main.py --step all

# 3. Visualiser les résultats
cat data/output/baseline_results.csv
```

Durée d'exécution : **1 à 3 minutes** selon la taille des données.

