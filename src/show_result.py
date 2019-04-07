import sys
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

def show_result(x, y_dict, model_name=''):
    fig, ax = plt.subplots()

    for key,val in y_dict.items():
        ax.plot(x, np.array(val)*100, label=key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy(%)")
    ax.legend()
    ax.grid()
    ax.set_title("Actvation function comparision ({})".format(model_name))
    plt.show()


def main(args):
    source = {}
    with open(args.file_name, 'r') as f:
        source = json.load(f)
    pprint(list(zip(source['y_dict'].keys(), [max(i) for i in source['y_dict'].values()])))
    show_result(source['x'], source['y_dict'], source['title'])


def get_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", help="your json file name", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()