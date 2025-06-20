import argparse
from utils.utils import model_params
from distributed_training import myDDP

def main():
    parser = argparse.ArgumentParser(description="Print model parameters from a checkpoint file.")
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    args = parser.parse_args()

    model_params(args.checkpoint)

if __name__ == "__main__":
    main()
