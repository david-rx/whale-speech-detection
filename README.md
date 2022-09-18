# whale-speech-detection

A deep-learning detector for Antarctic Baleen Whale sounds.

Uses data from "An open access dataset for developing automated detectors of Antarctic baleen whale sounds..." https://www.nature.com/articles/s41598-020-78995-8

--- WORK IN PROGRESS ---

Hypothesis is to use the YOHO algorithm https://www.mdpi.com/2076-3417/12/7/3293 to classify audio segments with presence of whale call types.

The first audio features will be will be mel-spectrogram and will be processed by a CNN or ViT.

After the prototype, the plan will be to train a self-supervised model based on https://ai.facebook.com/research/publications/conformer-based-self-supervised-learning-for-non-speech-audio-tasks/ or HuBERT on the larger dataset of unlabeled audio.
This will become the base of the supervised training and hopefully serve as a foundation model

## Tasks

### TO DO:
CNN setup (in progress)

ViT setup

HuBERT modification (design needed)

HuBERT setup

### Completed:
Parse Antartic Whale Dataset

Audio Features setup (mel spectrogram -- parameters to be iterated)

YOHO labels setup
