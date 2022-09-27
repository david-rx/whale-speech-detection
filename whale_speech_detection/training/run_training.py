import numpy as np
import random
# import configargparse
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch
from whale_speech_detection.data.utils import DEFAULT_FMAX, DEFAULT_FMIN, DEFAULT_HOP_LENGTH, DEFAULT_N_FFT

from whale_speech_detection.models.yoho_cnn.yoho_cnn import YohoCnn
from whale_speech_detection.data.load_dataset import CallDataset, load_dataset
from transformers import Trainer, ViTFeatureExtractor, ViTForImageClassification, Wav2Vec2ConformerForSequenceClassification
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerModel

from transformers import TrainingArguments
from PIL import Image
import librosa.display

# DATA_PATH = '/Users/david/downloads/AcousticTrends_BlueFinLibrary/BallenyIslands2015'
# EVAL_DATA_PATH = '/Users/david/downloads/AcousticTrends_BlueFinLibrary/RossSea2014'
DATA_PATH = "/home/david/Datasets/AcousticTrends_BlueFinLibrary/BallenyIslands2015"
EVAL_DATA_PATH = "/home/david/Datasets/AcousticTrends_BlueFinLibrary/BallenyIslands2015"

BATCH_SIZE = 16
PRETRAINED_VIT = "google/vit-base-patch16-224"

NUM_LABELS = 9
ID2LABEL = {} # TODO: define these
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt



class ViTDataCollator:

    def __init__(self, feature_extractor: ViTFeatureExtractor) -> None:
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        print(batch)
        inputs = [torch.abs(t[0]).cpu().numpy().astype(np.uint8) for t in batch]
        labels = [t[1] for t in batch]
        print(inputs)
        features = self.feature_extractor(inputs, return_tensors="pt")
        features["labels"] = labels
        return features

class Wav2VecConformerDataCollator:
    def __init__(self, feature_extractor: ViTFeatureExtractor, sampling_rate: int) -> None:
        self.feature_extractor = feature_extractor
        self.sampling_rate = sampling_rate

    def __call__(self, batch):
        inputs = [t[0] for t in batch]
        labels = [t[1] for t in batch]
        features = self.feature_extractor(inputs, return_tensors="pt", sampling_rate = self.sampling_rate)
        features["labels"] = labels
        return features



def train(vit: bool) -> None:
    """
    WIP! Loop not finished, optimized, run or tested.
    Currently uses data from
    a single monitoring site.
    """
    train_dataset = load_dataset(DATA_PATH, 'train')
    eval_dataset = load_dataset(DATA_PATH, 'eval')
    if vit:
        model = ViTForImageClassification.from_pretrained(PRETRAINED_VIT)
        fe =  ViTFeatureExtractor.from_pretrained(PRETRAINED_VIT)
    else:
        model = YohoCnn()
        fe = lambda x: x
    # model = torch.nn.Module()
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    optimizer = Adam(model.parameters, lr=5e-5)
    for batch in train_dataloader:
        yoho_inputs, yoho_labels = batch
        model_inputs = fe(yoho_inputs)
        loss = model(model_inputs, yoho_labels)
        loss.backwards()
        optimizer.step()

def train_hf_trainer(visualize = False, wav2vec_conformer: bool = True):
    """
    Train with the transformers trainer (deepspeed support.)
    Use for ViT.
    """
    train_dataset = load_dataset(DATA_PATH, 'train', raw_audio = wav2vec_conformer)
    eval_dataset = load_dataset(DATA_PATH, 'eval', raw_audio = wav2vec_conformer)
    if visualize:
        visualize_some_examples(train_dataset)


    if not wav2vec_conformer:
        model = ViTForImageClassification.from_pretrained(
        PRETRAINED_VIT,
        num_labels = NUM_LABELS,
        ignore_mismatched_sizes=True
        # id2label = ID2LABEL
        )
        feature_extractor =  ViTFeatureExtractor.from_pretrained(PRETRAINED_VIT)
        collator = ViTDataCollator(feature_extractor)
    else:
        feature_extractor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        model = Wav2Vec2ConformerForSequenceClassification.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft",
        num_labels = NUM_LABELS,
        ignore_mismatched_sizes=True)
        collator = Wav2VecConformerDataCollator(feature_extractor, 16000)

    training_args = TrainingArguments(
        output_dir="./whale-speech-detector-ViT",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics) 
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

def visualize_some_examples(dataset: CallDataset, num_images_to_show = 5):
    for i in range(num_images_to_show):
        example_idx = random.randint(0, len(dataset) - 1)
        example = dataset[example_idx]
        print(example[0])
        print(example[0].shape)
        print(example[1])
        print(example[1].shape)

        example_image = example[0]
        example_label = example[1]
        # img = Image.fromarray(example_image.cpu().numpy())
        # img.show(title = str(example_label))

        fig, ax = plt.subplots()
        lib_img = librosa.display.specshow(example_image.cpu().numpy(), x_axis = 'time', y_axis = 'mel', sr = 1000, ax=ax, hop_length = DEFAULT_HOP_LENGTH, n_fft = DEFAULT_N_FFT, win_length=400, fmin=DEFAULT_FMIN, fmax = DEFAULT_FMAX)
        fig.colorbar(lib_img, ax=ax, format='%+2.0f dB')

        ax.set(title=example_label)
        plt.show()

    for i in range(num_images_to_show):
        valid_indices = [i for i in range(len(dataset)) if 1 in dataset[i][1]]
        print("there are {} examples with non-zero labels".format(len(valid_indices)))
        example_idx = random.choice(valid_indices)
        example = dataset[example_idx]
        print(example[0])
        print(example[0].shape)
        print(example[1])
        print(example[1].shape)

        example_image = example[0]
        example_label = example[1]
        # img = Image.fromarray(example_image.cpu().numpy())
        # img.show(title = str(example_label))

        fig, ax = plt.subplots()
        lib_img = librosa.display.specshow(example_image.cpu().numpy(), x_axis = 'time', y_axis = 'mel', sr = 1000, ax=ax, hop_length = DEFAULT_HOP_LENGTH, n_fft = DEFAULT_N_FFT, win_length=400, fmin=DEFAULT_FMIN, fmax = DEFAULT_FMAX)
        fig.colorbar(lib_img, ax=ax, format='%+2.0f dB')

        ax.set(title=example_label)
        plt.show()











if __name__ == '__main__':
    # p = configargparse.ArgParser()
    train_hf_trainer(visualize = False)