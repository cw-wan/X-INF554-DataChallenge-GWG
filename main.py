from utils.training import MODELS
from utils.tuning_roberta import train_roberta, compute_embeddings
import argparse

parser = argparse.ArgumentParser(description='Project Enter Point',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--selected_model', '-s', help='model to select')
parser.add_argument('--mode', '-m', help='train or test', required=True)
parser.add_argument('--epoch', '-e', help='for pre-compute')
args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        MODELS[args.selected_model]['train']()
    elif args.mode == "test":
        MODELS[args.selected_model]['test']()
    elif args.mode == "tune":
        train_roberta()
    elif args.mode == "compute":
        compute_embeddings(total_epoch=int(args.epoch))
