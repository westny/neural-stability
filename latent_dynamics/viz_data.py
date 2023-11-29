import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from latent_dynamics.dm_latent import CustomHDF5Dataset


if __name__ == "__main__":
    data = {}
    real_names = {"spring": "Spring-Mass System",
                  "twoXpendulum": "Double Pendulum",
                  "mujoco": "3D Room (MuJoCo)",
                  "molecules": "Molecular Dynamics (16 particles)"}

    # 2, 99
    for name, seed in zip(["spring", "twoXpendulum", "mujoco", "molecules"], [42, 99, 2, 3]):
        seed_everything(seed)
        file_path = f"../data/{name}/test.hdf5"
        ds = CustomHDF5Dataset(file_path, sampling_time=0.15)
        dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=1)

        for batch in tqdm(dl):
            inp, _ = batch
            break

        data[name] = inp

    nrows = len(data)

    grey = torch.tensor([0.2989, 0.5870, 0.1140])[:, None]

    # plot 10 samples from each dataset
    fig, axes = plt.subplots(nrows=nrows, ncols=10, figsize=(10, nrows))
    for i, (name, inp) in enumerate(data.items()):
        for j in range(10):
            if name == "spring":
                axes[i, j].imshow(inp[0, j].permute(1, 2, 0) @ grey, cmap='gray')
            else:
                axes[i, j].imshow(inp[0, j].permute(1, 2, 0))
            axes[i, j].axis('off')

        # Create a dummy subplot for the title
        row_title = fig.add_subplot(nrows, 1, i + 1)
        row_title.set_title(real_names[name], fontsize=12)
        row_title.axis('off')  # Hide the subplot

    plt.tight_layout()
    plt.savefig("latent_dyn.png", dpi=300)
    plt.show()
