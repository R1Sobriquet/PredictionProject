## ** Architecture du projet et dictionnaire de données**

### **Architecture du projet**
```
forecasting_project/
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py      # Ingestion et nettoyage
│   ├── data_processing.py     # Enrichissement des données
│   ├── visualization.py       # Graphiques et analyses
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py        # Modèles de référence
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration
├── tests/
│   ├── __init__.py
│   ├── test_data_ingestion.py
│   └── test_data_processing.py
├── data/
│   ├── raw/                   # Données brutes
│   ├── processed/             # Données nettoyées
│   └── output/                # Résultats
├── docs/
│   └── data_dictionary.md
├── requirements.txt
└── main.py
```

### **Dictionnaire de données**

| Champ | Type | Description | Utilité pour la prévision                       |
|-------|------|-------------|-------------------------------------------------|
| `date_ligne_commande` | Date | Date de la commande (format YYYY-MM-DD) | **ESSENTIEL** - Variable temporelle principale  |
| `id_article` | Integer | Identifiant unique de l'article | **ESSENTIEL** - Permet de regrouper par article |
| `ref_article` | String | Référence métier de l'article (ex: REF001) | OPTIONNEL - Pour la lisibilité                  |
| `quantite` | Integer | Quantité commandée | **ESSENTIEL** - Variable cible à prédire        |

