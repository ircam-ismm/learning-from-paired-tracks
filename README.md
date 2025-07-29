# Learning Relationships between Separate Audio Tracks for Creative Applications

This project explores learning-based approaches to model relationships between separate audio tracks, enabling creative applications such as symbolic generation and guided audio synthesis.

This repository contains the official codebase accompanying the AIMC2025 paper:

**"Learning Relationships between Separate Audio Tracks for Creative Applications"**  
*Bujard et al., 2025*

Audio examples can be found at https://ircam-ismm.github.io/MoisesDB-audio-examples/

---

## Overview

This repository contains:
- MusicDataset : The dataset structures for loading, pre-processing and batching audio.
- architecture : The pytorch Module classes for the Perception (aka Encoder), Decision and Seq2Seq (for auto-completion and coupling tasks) modules.
- utils : a utilitarity folder conatining dataset folder generation scripts, metrics for evaluation, and most important the dicy2_generator.py script containing all functions relevant to the Action module and exploitation of trained Decision module (generating symbolic specifications, converting symbolic specifications into audio).
- training, evaluation, quantization scripts.

The repository includes all the code necessary to reproduce the results presented in the paper, **except** for:

- The **pretrained Wav2Vec 2.0 model** trained on music, introduced in *Ragano et al., 2023*.  
  → Please contact the authors of that paper directly to obtain access to the model weights.

- The **MICA dataset**, which is proprietary and not publicly available.  
  → As a result, only experiments using **MoisesDB** can be reproduced with the current repository.

---

## Tutorials

To facilitate usage, three tutorial scripts are provided:

1. **`train_model.py`**
   Train the Decision module on a pair of audio tracks.

2. **`use_decision.py`**
   Generate a symbolic specification (e.g., structure or timing) from an audio input using a trained Decision module.

3. **`generate_audio.py`**
   Given a *guide* track, a *memory* track, and a trained model, this script generates a response audio track conditioned on the guide.

---
## Installation
- python 3.10
- install requirements with "pip install -r requirements.txt"
- install Dicy2-python library from "https://github.com/DYCI2/Dicy2-python" and place it outside this folder, i.e. ../Dicy2-python.
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
