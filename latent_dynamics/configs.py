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
