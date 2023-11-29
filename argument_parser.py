from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    elif value.lower() in false_vals:
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Neural stability arguments')

# Program arguments
parser.add_argument('--main-seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--scnd-seed', type=int, default=None,
                    help='re-random seed. Used for different data splits and teacher generation.')
parser.add_argument('--n-workers', type=int, default=4,
                    help='number of workers in dataloader')
parser.add_argument('--use-logger', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if logger should be used')
parser.add_argument('--use-cuda', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if cuda exists and should be used')
parser.add_argument('--store-data', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if checkpoints should be stored')
parser.add_argument('--overwrite', type=str_to_bool, default=True,
                    const=True, nargs="?", help='overwrite if model exists (default: True)')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to save name')
parser.add_argument('--dry-run', type=str_to_bool, default=True,
                    const=True, nargs="?", help='verify the code and the model')
parser.add_argument('--pin-memory', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if the data should be pinned to memory')
parser.add_argument('--persistent-workers', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if the workers should be persistent')
parser.add_argument('--stability-init', type=str_to_bool, default=False,
                    help='init parameters within solver stability region (default: False)')
parser.add_argument('--config', type=str, default="MTS_engine",
                    help='config file path for experiment (default: MTS_engine)')

args = parser.parse_args()
