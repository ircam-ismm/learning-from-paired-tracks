# Learning Relationships between Separate Audio Tracks for Creative Applications

This repository contains the official codebase accompanying the AIMC2025 paper:

**"Learning Relationships between Separate Audio Tracks for Creative Applications"**  
*Bujard et al., 2025*

---

## Overview

This project explores learning-based approaches to model relationships between separate audio tracks, enabling creative applications such as symbolic generation and guided audio synthesis.

The repository includes all the code necessary to reproduce the results presented in the paper, **except** for:

- The **pretrained Wav2Vec 2.0 model** trained on music, introduced in *Ragano et al., 2023*.  
  → Please contact the authors of that paper directly to obtain access to the model weights.

- The **MICA dataset**, which is proprietary and not publicly available.  
  → As a result, only experiments using **MoisesDB** can be reproduced with the current repository.

---

## Tutorials

To facilitate usage, three tutorial scripts are provided:

1. **`train_model.py`**  TODO : finish testing and debugging
   Train the Decision module on a pair of audio tracks.

2. **`use_decision.py`**  TODO : create script
   Generate a symbolic specification (e.g., structure or timing) from an audio input using a trained Decision module.

3. **`generate_audio.py`**  TODO : finish testing and debugging
   Given a *guide* track, a *memory* track, and a trained model, this script generates a response audio track conditioned on the guide.

---

## Citation

If you use this code in your work, please cite:

```bibtex
@inproceedings{bujard2025relationships,
  title={Learning Relationships between Separate Audio Tracks for Creative Applications},
  author={Bujard Balthazar, Nika Jérôme, Obin Nicolas, Bevilacqua Frédéric},
  booktitle={Proceedings of the 6th Conference on AI Music Creativity (AIMC 2025)},
  year={2025}
}
