# whale-speech-detection

A deep-learning detector for Antarctic Baleen Whale sounds.

Objective is to first create a detector with immediate use for passive acoustic monitoring. Then, create a general foundation model using self-supervised learning on the larger unlabeled segment of whale call data.

Uses data from "An open access dataset for developing automated detectors of Antarctic baleen whale sounds..." https://www.nature.com/articles/s41598-020-78995-8

--- WORK IN PROGRESS ---

Hypothesis is to use the YOHO algorithm https://www.mdpi.com/2076-3417/12/7/3293 to classify audio segments with presence of whale call types.

The first audio features will be will be mel-spectrogram feeding a CNN or ViT. Will try both training a small model from scratch vs.
a larger model pretrained on ImageNet.

After benchmarking this model, the plan is to train self-supervised model based on this conformer + Wav2vec 2.0 model https://ai.facebook.com/research/publications/conformer-based-self-supervised-learning-for-non-speech-audio-tasks/ or based on HuBERT on the larger dataset of unlabeled audio.
This will become the base of the supervised training, then hopefully serve as a foundation model for other marine audio tasks.

## Tasks

### TO DO:
CNN setup and training (in progress)

ViT setup and training

Self-supervised base model setup and training

### Completed:
Parse Antartic Whale Dataset

Audio Features setup (mel spectrogram -- parameters to be iterated)

YOHO labels setup
