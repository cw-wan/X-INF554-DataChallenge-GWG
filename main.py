from utils.training import MODELS
import argparse

parser = argparse.ArgumentParser(description='Project Enter Point',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--selected_model', '-s', help='model to select', required=True)
parser.add_argument('--mode', '-m', help='train or test', required=True)
parser.add_argument('--load_epoch', '-e', help='if test, which checkpoint to load')
args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        MODELS[args.selected_model]['train']()
    else:
        MODELS[args.selected_model]['test'](load_epoch=args.load_epoch)
