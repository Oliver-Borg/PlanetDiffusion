from utils import PlanetConfig, image_grid
from dataset import RAMDataset

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from diffusers.models import AutoencoderKL
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sample(vae, name: str):
    planet_cfg = PlanetConfig()
    dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = 100
    grid = []
    row_labels = []
    t_to_im = lambda t: Image.fromarray((t.squeeze(0).permute(1, 2, 0).cpu().numpy() *127.5 + 127.5).astype(np.uint8))
    for i, item in enumerate(tqdm(dataloader)):
        if i >= rows:
            break
        target = item['target_image'].to(device)
        with torch.no_grad():
            target_sat = target[:, :3]
            target_dem = target[:, 3:].repeat(1, 3, 1, 1)
            latent_sat = vae.encode(target_sat).latent_dist.sample()
            latent_dem = vae.encode(target_dem).latent_dist.sample()
            output_sat = vae.decode(latent_sat).sample.clamp(-1, 1)
            output_dem = vae.decode(latent_dem).sample.clamp(-1, 1)
            row = [output_sat, target_sat, output_dem, target_dem]
            sat_loss = torch.nn.functional.mse_loss(output_sat, target_sat)
            dem_loss = torch.nn.functional.mse_loss(output_dem, target_dem)
            row_labels.append(f"{sat_loss.item():.4f}, {dem_loss.item():.4f}")
            row = list(map(t_to_im, row))
            grid.extend(row)  
    grid = image_grid(grid, rows, 4, row_labels=row_labels)
    grid.save(f"tests/TestAutoEncoder-{name}.png")


repo_ids = [
    "stabilityai/stable-diffusion-2-base",
    "sstabilityai/stable-diffusion-2-1",
    
]
repo_id = "stabilityai/stable-diffusion-2-base"
vae = AutoencoderKL.from_pretrained(repo_id, subfolder='vae').vae.to(device)
sample(vae, "stable")