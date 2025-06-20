This repository contains the code from the AIMC2025 paper "Learning Relationships betweenSeparate Audio Tracks for Creative Applications" (Bujard et al. 2025).
All the code for reproducing results is available, except for the weights of the Wav2Vec2.0 pre-trained on music from "Learning music representations with wav2vec 2.0" (Ragano et al, 2023) which should be askedf to the authors of the paper.
Since MICA dataset is proprietary data, only MoisesDB results can be reproduced.

3 tutorial files are provided for an easy use of the code:
1) train_model.py : used to train the decision module on one pair of tracks.
2) use_decision.py : generates a symbolic specification from a trained Decision module given an audio track.
3) generate_audio.py : given a guide track, a memory track and a trained model, this script generates audio in response of the guide track
