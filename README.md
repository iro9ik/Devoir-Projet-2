# 🎨 Devoir Projet 2 : Modèles Génératifs avec PyTorch

Ce projet explore les architectures génératives profondes sur le jeu de données manuscrit MNIST.

---

## 📂 Contenu du Dossier

- **`Modeles_generatifs_pytorch.ipynb`** : Le notebook pratique implémentant les modèles génératifs.
- **`Modeles_generatifs_pytorch_avec_explications.ipynb`** : Le notebook détaillé avec les explications théoriques bloc par bloc.
- **`pyproject.toml`** : Fichier de configuration de l'environnement Python 3.11 avec UV.
- **`.gitignore`** : Fichier ignorant le cache MNIST et l'environnement virtuel.

---

## 📝 Description des Sections

### 1. 🎯 Modèle Discriminatif (Baseline)
- Modèle de classification standard apprenant $P(y \mid x)$ pour reconnaître le chiffre de l'image.

### 2. 🗜️ Auto-Encodeur Déterministe (AE)
- Compresse l'image dans un espace latent 2D puis la reconstruit avec une perte MSE.
- Permet de visualiser la séparation des chiffres sur un plan 2D, mais présente des zones vides.

### 3. 🎲 Auto-Encodeur Variationnel (VAE)
- Remplace les points déterministes par une distribution probabiliste normale $\mathcal{N}(0, I)$.
- Utilise l'astuce de reparamétrisation ($z = \mu + \sigma \odot \epsilon$) pour permettre la descente de gradient.
- Permet de générer de nouveaux chiffres continus en explorant la grille latente 2D.

### 4. ⚔️ Réseaux Antagonistes Génératifs (GAN)
- Jeu minimax entre un Générateur qui crée de faux chiffres et un Discriminateur qui les évalue.

### 5. 🌫️ Modèles de Diffusion (DDPM)
- Processus de bruitage progressif des images puis apprentissage du débruitage étape par étape.

---

## 🚀 Comment Exécuter

1. Configurez l'environnement avec UV :
   ```bash
   uv sync
   ```
   *Ou avec pip :*
   ```bash
   pip install torch torchvision matplotlib torcheval
   ```
2. Lancez le notebook dans VS Code pour visualiser les reconstructions et générations de chiffres.
