from importlib.resources import path
from custom_image_binary_v1 import run
import argparse


def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--valiation_data_path', type=str, required=True)

if __name__ == "__main__":
    # def run(TRAIN_DATA_PATH = "D:/code/py/train", VALIDATION_DATA_PATH = "D:/code/py/test",BATCH_SIZE = 8, LR_RATE=0.001, EPOCH = 10):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, required=False, default=1)
    parser.add_argument('--batch_size', type=int, required=False, default=8)
    parser.add_argument('--lr', type=float, required=False, default=0.001)
    parser.add_argument('--train_data_path', type=str, required=False, default='D:/code/py/train')
    parser.add_argument('--validation_data_path', type=str, required=False, default='D:/code/py/test')
    
    args = parser.parse_args()
    run(TRAIN_DATA_PATH = args.train_data_path, VALIDATION_DATA_PATH = args.validation_data_path,BATCH_SIZE = args.batch_size, LR_RATE=args.lr, EPOCH = args.num_epoch)