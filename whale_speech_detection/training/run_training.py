import configargparse
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
import torch

from whale_speech_detection.models.yoho_cnn.yoho_cnn import YohoCnn
from whale_speech_detection.data.load_dataset import load_dataset


DATA_PATH = '/Users/david/downloads/AcousticTrends_BlueFinLibrary/BallenyIslands2015'
EVAL_DATA_PATH = '/Users/david/downloads/AcousticTrends_BlueFinLibrary/RossSea2014'
BATCH_SIZE = 16

def train() -> None:
    train_dataset = load_dataset(DATA_PATH, 'train')
    eval_dataset = load_dataset(DATA_PATH, 'eval')
    model = YohoCNN()
    # model = torch.nn.Module()
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
    optimizer = Adam(model.parameters, lr=10^-5)
    for batch in train_dataloader:
        yoho_inputs, yoho_labels = batch
        loss = model(yoho_inputs, yoho_labels)
        loss.backwards()
        optimizer.step()

    

if __name__ == '__main__':
    # p = configargparse.ArgParser()
    train()