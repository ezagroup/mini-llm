# Mini LLM - Petit Modèle de Langage Depuis Zéro

## Objectif

Ce projet vise à construire **progressivement** un petit modèle de langage en Python et PyTorch, **entièrement depuis zéro**, sans dépendre d'aucun modèle pré-entraîné ni d'API d'IA externe.

L'objectif final est d'intégrer ce modèle dans un bot Discord, mais la première étape consiste à comprendre et implémenter chaque composant du pipeline complet.

## Principes Fondamentaux

- ✅ **Pas d'API externe** : Pas de ChatGPT, Claude, Gemini ou autre IA cloud
- ✅ **Pas de modèle pré-entraîné** : Construction depuis zéro
- ✅ **Apprentissage progressif** : Chaque étape dépend de la précédente
- ✅ **Ressources limitées** : Conçu pour être testé sur une machine standard
- ✅ **Architecture extensible** : Prête à évoluer vers une machine plus puissante

## Feuille de Route

Le projet suivra ces étapes principales :

1. **Préparation des données** → Collecter et charger des textes bruts
2. **Tokenization** → Convertir du texte en tokens numériques
3. **Dataset** → Préparer des séquences d'entraînement
4. **Modèle Transformer** → Implémenter une petite architecture Transformer
5. **Entraînement** → Boucle d'entraînement et sauvegarde des checkpoints
6. **Génération de texte** → Générer du texte avec le modèle entraîné
7. **Bot Discord** → Intégrer le modèle dans un bot Discord

## Structure du Projet

```
mini-llm/
├── data/              # Données brutes et traitées
├── src/
│   ├── tokenizer/     # Tokenization des textes
│   ├── dataset/       # Chargement et préparation des données
│   ├── model/         # Architecture Transformer
│   ├── train/         # Boucle d'entraînement
│   ├── generate/      # Génération de texte
│   └── utils/         # Utilitaires communs
├── scripts/           # Scripts d'exécution
├── notebooks/         # Exploration et expérimentation
├── checkpoints/       # Modèles entraînés (vides au départ)
├── README.md          # Ce fichier
├── requirements.txt   # Dépendances Python
└── .gitignore        # Fichiers à ignorer par Git
```

## Prérequis

- Python 3.9+
- PyTorch
- pip ou conda pour la gestion des dépendances

## Installation

```bash
git clone https://github.com/ezagroup/mini-llm.git
cd mini-llm
pip install -r requirements.txt
```

## État Actuel

Cette version initiale contient uniquement la structure du projet. Le code implémentant les composants sera ajouté progressivement.

## Licence

MIT
