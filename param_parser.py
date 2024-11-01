"""Parameter parsing."""

import argparse
import torch
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the Cora dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run MixHop.")

    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument("--saveresults",
                        type=str,
                        default=True)

    parser.add_argument("--path_weight",
                        nargs="?",
                        default="/home/miaorui/MGCN/weights/",
                        help="path_weight.")

    parser.add_argument("--path_result",
                        nargs="?",
                        default="/home/miaorui/MGCN/results/",
                        help="path_result.")

    parser.add_argument("--dataset_name",
                        nargs="?",
                        default="USA",
	                help="dataset_name:Farm、USA、Santa、Bay")

    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
	                help="Number of training epochs. Default is 1000.")

    parser.add_argument("--early-stopping",
                        type=int,
                        default=10,
	                help="Number of early stopping rounds. Default is 10.")

    parser.add_argument("--train_ratio",
                        type=float,
                        default=0.05,
	                help="Training set ratio. Default is 0.5%.")

    parser.add_argument("--val_ratio",
                        type=float,
                        default=0.01,
	                help="Validation set ratio. Default is 1%.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for train-test split. Default is 42.")

    parser.add_argument("--n_segments_init",
                        type=int,
                        default=math.ceil(307*241/4),
                        help="n_segments_init. Default is Farm:420*140/5=11760,USA:307*241/4=14797,Santa:984*740/250=2913,Bay:600*500/200=1500,River:463*241.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01,
	                help="Learning rate. Default is 0.01.")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="weight_decay. Default is 0.01.")

    parser.add_argument("--layers-1",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space (top). E.g. 200 20.")

    parser.add_argument("--layers-2",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space (bottom). E.g. 200 200.")


    parser.set_defaults(layers_1=[128, 128, 32])
    parser.set_defaults(layers_2=[64, 64, 8])
    
    return parser.parse_args()
