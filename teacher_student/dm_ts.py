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

from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from teacher_student.reference_system import *


class LitDataModule(LightningDataModule):
    m: MultiStateTeacherNetwork | LinearMultiStateTeacherNetwork

    def __init__(self, args, config: dict) -> None:
        super().__init__()
        self.batch_size = config['batch_size']
        self.seed = config['data_seed']
        self.dataset = config['name']
        self.duration = config['duration']
        self.sample_time = config['sample_time']
        self.batch_time = config['batch_time']
        self.ctrl_sig = config['control_signal']
        self.tst_sig = config['test_signal']

        self.n_workers = args.n_workers

        if self.dataset == "nonlinear":
            self.m = MultiStateTeacherNetwork(config['teacher'], self.seed)
        else:
            self.m = LinearMultiStateTeacherNetwork(config['teacher'])

        self.ds_train = NeuralStabilityDataset(self.m,
                                               'training',
                                               self.duration,
                                               self.ctrl_sig,
                                               self.batch_time,
                                               self.sample_time)

        self.ds_test = NeuralStabilityDataset(self.m,
                                              'testing',
                                              self.duration,
                                              self.tst_sig,
                                              self.batch_time,
                                              self.sample_time)

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=self.ds_train.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=self.ds_test.collate_fn)

    def test_dataloader(self):
        dataset = NeuralStabilityDataset(self.m,
                                         'testing',
                                         self.duration,
                                         self.ctrl_sig,
                                         self.batch_time,
                                         self.sample_time)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          collate_fn=dataset.collate_fn)


class NeuralStabilityDataset(Dataset):
    def __init__(self,
                 reference_model,
                 train_val_test: str = "training",
                 duration: float = 30,
                 reference_input: str = "pwm_sine",
                 batch_time: int = 0,
                 sample_time: float = 0.01):
        self.poles = None
        self.data, self.t, self.input = self._create_data(reference_model,
                                                          duration,
                                                          reference_input,
                                                          sample_time,
                                                          train_val_test)
        self.y0 = self.data[0:1, :]
        self.data_size = len(self.t)
        self.b_time = self.data_size - 1 if batch_time == 0 else batch_time
        self.inp_idx = torch.arange(max(self.data_size - self.b_time, 1), dtype=torch.int64)

    def _create_data(self, teacher, duration=30., ref_inp="pwm_sine", step_size=0.01, train_val_test="training"):
        t_start = 0.
        t_end = duration

        n_sys_states = teacher.n_states
        true_y0 = torch.rand(1, n_sys_states)
        retries = 1
        while retries <= 5:
            try:
                true_y, time, inp = teacher.generate_data(true_y0, t_start, t_end, step_size, ref_inp)
                # assert torch.isnan(true_y).any() is False
            except AssertionError:
                print(
                    f"Error in data generation. Will retry by re-initializing parameter values. Attempt #{retries}")
                retries += 1
                break
                # m.reset_parameters()
            else:
                retries = 0
                break

        if retries > 0:
            raise RuntimeError("Unable to automatically solve data generation error. Aborting.")

        true_y = true_y.view(true_y.shape[0], n_sys_states)

        try:
            reals = []
            imags = []

            for i in range(true_y.shape[0] - 1):
                eigs = torch.linalg.eigvals(teacher.state_jacobian(true_y[i:i + 1], inp[i:i + 1]))
                reals.append(eigs.real.detach())
                imags.append(eigs.imag.detach())

            reals = torch.cat(reals, dim=0)
            imags = torch.cat(imags, dim=0)

            self.poles = torch.stack((reals, imags), dim=0)
        except AttributeError:
            eig = torch.linalg.eigvals(teacher.state_mat())
            reals, imags = eig.real, eig.imag
            self.poles = torch.stack((reals, imags), dim=0)

        if train_val_test == "validation":
            breakpoint()

        return true_y, time, inp

    def __len__(self):
        return len(self.inp_idx)

    @staticmethod
    def collate_fn(batch):
        y = torch.stack([b[0] for b in batch], dim=1)
        i = torch.stack([b[1] for b in batch], dim=1)
        t = batch[0][-1]
        return y, i, t

    def __getitem__(self, idx):
        s = self.inp_idx[idx]
        t = self.t[:self.b_time]  # (T)
        y = self.data[s: s + self.b_time]  # (T, M, D)
        inp = self.input[s: s + self.b_time]  # (T, M, D)
        return y, inp, t
