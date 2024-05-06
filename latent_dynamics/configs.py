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

config_mass_spring = {"dx_dt": ("float", (512, 2)),
                      "x": ("float", (512, 2)),
                      "other/offset": ("int", (1, 1)),
                      "other/k": ("float", (1,)),
                      "other/color_index": ("int", (1, 1)),
                      "other/m": ("float", (1,)),
                      "image": ("uint8", (512, 32, 32, 3))}

config_pendulum = {"dx_dt": ("float", (512, 2)),
                   "x": ("float", (512, 2)),
                   "other/offset": ("float", (1, 2)),
                   "other/l": ("float", (1,)),
                   "other/g": ("float", (1,)),
                   "other/color_index": ("int", (1, 1)),
                   "other/m": ("float", (1,)),
                   "image": ("uint8", (512, 32, 32, 3))}

config_double_pendulum = {"dx_dt": ("float", (512, 4)),
                          "x": ("float", (512, 4)),
                          "other/offset": ("float", (1, 2)),
                          "other/l_1": ("float", (1,)),
                          "other/l_2": ("float", (1,)),
                          "other/g": ("float", (1,)),
                          "other/color_index": ("int", (1, 2)),
                          "other/m_1": ("float", (1,)),
                          "other/m_2": ("float", (1,)),
                          "image": ("uint8", (512, 32, 32, 3))}

config_two_bodies = {"dx_dt": ("float", (512, 8)),
                     "x": ("float", (512, 8)),
                     "other/offset": ("float", (1, 2)),
                     "other/g": ("float", (1,)),
                     "other/color_index": ("int", (1, 2)),
                     "other/m": ("float", (1,)),
                     "image": ("uint8", (512, 32, 32, 3))}

config_mujoco_room = {"dx_dt": ("float", (256, 6)),
                      "x": ("float", (256, 6)),
                      "image": ("uint8", (256, 32, 32, 3))}

config_molecules = {"dx_dt": ("float", (256, 32)),
                    "x": ("float", (256, 32)),
                    "image": ("uint8", (256, 32, 32, 3))}
