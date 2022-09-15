# whale-speech-detection

A deep-learning detector for Antarctic Baleen Whale sounds.

Uses data from "An open access dataset for developing automated detectors of Antarctic baleen whale sounds..." https://www.nature.com/articles/s41598-020-78995-8

--- WORK IN PROGRESS ---

Hypothesis is to use the YOHO algorithm https://www.mdpi.com/2076-3417/12/7/3293 to classify audio segments with presence of whale call types.

The first audio features will be will be MFCC and will be processed by a CNN or ViT. 

After the prototype, the plan will be to train a self-supervised model based on HuBERT on the larger dataset of unlabeled audio. 

## Tasks 

### TO DO:
CNN setup (in progress)

ViT setup   

HuBERT modification (design needed)

HuBERT setup

### Completed:
Parse Antartic Whale Dataset

Audio Features setup (MFCC)

YOHO labels setup
