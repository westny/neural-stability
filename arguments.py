# Copyright 2024, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value: bool | str) -> bool:
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    if value.lower() in false_vals:
        return False
    raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Neural stability arguments')

# Program arguments
parser.add_argument('--main-seed', type=int, default=1234,
                    help='random seed (default: 1234)')
parser.add_argument('--scnd-seed', type=int, default=None,
                    help='Used for different data splits and teacher generation. Default: None')
parser.add_argument('--n-workers', type=int, default=1,
                    help='number of workers in dataloader (default: 1)')
parser.add_argument('--use-logger', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='if logger should be used (default: False)')
parser.add_argument('--use-cuda', type=str_to_bool, default=False,
                    const=True, nargs="?",
                    help='if cuda exists and should be used (default: False)')
parser.add_argument('--store-model', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if checkpoints should be stored (default: True)')
parser.add_argument('--overwrite', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='overwrite if model exists (default: True)')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to save name (default: "")')
parser.add_argument('--dry-run', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='verify the code and the model (default: True)')
parser.add_argument('--pin-memory', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if the data should be pinned to memory (default: True)')
parser.add_argument('--persistent-workers', type=str_to_bool, default=True,
                    const=True, nargs="?",
                    help='if the workers should be persistent (default: True)')
parser.add_argument('--stability-init', type=str_to_bool, default=True,
                    help='init parameters within solver stability region (default: False)')
parser.add_argument('--config', type=str, default="engine",
                    help='config file path for experiment (default: engine)')

args = parser.parse_args()
