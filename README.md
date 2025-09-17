# Tennis Shot Classification with Video Transformers

This project implements a deep learning pipeline to automatically classify **tennis strokes** into *forehand* and *backhand* using state-of-the-art video models.  
The approach combines modern preprocessing, temporal modeling, and transfer learning on a curated dataset of high-quality annotated swings.  

---

## Project Overview

- **Goal**: Given a short video clip of a player’s swing, predict whether the shot is a **forehand** or a **backhand**.  
- **Model**: Fine-tuned [TimeSformer](https://arxiv.org/abs/2102.05095), a Transformer-based architecture for video understanding, pretrained on Kinetics-400.  
- **Pipeline**:
  1. **Data preprocessing**: group raw frames into swings (~10 frames each), resize, and normalize.
  2. **Training**: linear probe followed by full fine-tuning, with gradient checkpointing and mixed precision for efficient training on GPU (Google Colab T4).  
  3. **Evaluation**: accuracy measured on held-out validation and test sets.  

---

## Results

- **Validation accuracy**: up to **88.9%**  
- **Test accuracy**: **94.7%** on unseen swings  
- Demonstrates robust separation between forehand and backhand strokes, despite limited data (~120 swings total).

---

## Dataset

This project uses the Tennis Player Actions Dataset for Human Pose Estimation (Mendeley Data):

Citation:
Omar, A., Awad, W., & Nassar, A. (2020). Tennis Player Actions Dataset for Human Pose Estimation. Mendeley Data, V1.
https://data.mendeley.com/datasets/ftn47vw6c3/1

