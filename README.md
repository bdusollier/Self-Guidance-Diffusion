# Autoguidance (Self-Guidance) for Diffusion Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdusollier/Self-Guidance-Diffusion/blob/main/autoguidance_notebook.ipynb)

Ce repository contient un **notebook pédagogique** explorant la méthode de **Self-Guidance (Autoguidance)** pour les modèles de diffusion, inspirée des travaux récents de **NVIDIA (2024)**.

Le but est de comprendre **intuitivement et mathématiquement** comment le guidage améliore la génération, à travers un **toy example 2D** avant de généraliser aux modèles de diffusion classiques.

---

## Idée principale

Les modèles de diffusion génèrent des données en partant du bruit et en le supprimant progressivement à l’aide d’un **score network**.

L’autoguidage repose sur l’idée suivante :

> Comparer un **modèle expert** (bien entraîné) à un **modèle amateur** (sous-entraîné)  
> et utiliser leur différence pour **amplifier les bons gradients** lors du sampling.

Formule clé utilisée dans le notebook :

$$\text{Score}_{guided} = \text{Score}_{amateur} + w \cdot (\text{Score}_{expert} - \text{Score}_{amateur})$$

où `w` contrôle la force du guidage.

---

## Contenu du notebook

**`autoguidance_better.ipynb`** couvre :

1.  **Toy example en 2D**
   - Données synthétiques en forme de "S"
   - Visualisation des distributions et trajectoires de sampling

2.  **Deux modèles de score**
   - **Expert (`D₁`)** : entraîné longuement
   - **Amateur (`D₀`)** : entraîné peu de temps

3.  **Processus de diffusion inverse**
   - Sampling sans guidage
   - Sampling avec autoguidage
   - Comparaison visuelle des résultats

4.  **Étude de l’impact du poids de guidage**
   - Effet de `w` sur la qualité et la stabilité des échantillons

---

##  Concepts clés abordés

- Denoising Diffusion Models  
- Score Matching  
- SDE / ODE de diffusion inverse  
- Self-Guidance / Autoguidance  
- Interprétation géométrique du score  

---

## 🚀 Exécution

## Prérequis
- Python ≥ 3.8
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Modèles pré-entraînés

Des **modèles déjà entraînés** (`expert_ckpt.pth` et `amateur_ckpt.pth`) sont fournis dans le repository.  
Ils peuvent être chargés directement depuis le notebook afin d’éviter de relancer l’entraînement complet.


### Lancer le notebook
```bash
jupyter notebook autoguidance_better.ipynb


