from diffusers import AutoencoderKL
import torch
from planetAI.src.data.dataset import RAMDataset
from planetAI.src.data.utils import PlanetConfig
from tqdm import tqdm

num_workers = 4
batch_size = 12

torch.manual_seed(0)
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

repo_id = "stabilityai/stable-diffusion-2-base"
vae = AutoencoderKL.from_pretrained(repo_id, subfolder='vae').to(device)
size = 256

planet_cfg = PlanetConfig(data_dir='./planetAI/data')
dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

all_sat_latents = []
all_dem_latents = []
for item in tqdm(loader):
    y = item['target_image'].to(device)
    sat_targets = y[:, 0:3]
    dem_targets = y[:, 3:4].repeat(1, 3, 1, 1)
    sat_latents = vae.encode(sat_targets).latent_dist.sample()
    dem_latents = vae.encode(dem_targets).latent_dist.sample()
    latents = torch.cat([sat_latents, dem_latents], dim=1)
    all_sat_latents.append(sat_latents.cpu())
    all_dem_latents.append(dem_latents.cpu())

for all_latents in [all_sat_latents, all_dem_latents]:
    all_latents_tensor = torch.cat(all_latents)
    std = all_latents_tensor.std().item()
    normalizer = 1 / std
    print(f'{normalizer = }')