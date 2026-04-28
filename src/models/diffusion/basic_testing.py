from diffusers import DDPMPipeline
import torch
from PIL import Image
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pipeline = DDPMPipeline.from_pretrained('./models/diffusion-test/').to(device)

def make_grid(images):
    rows = int(sqrt(len(images)))
    while len(images) % rows != 0:
        rows -= 1
    cols = len(images) // rows

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

for steps in [25]:
    seed = 42
    for n in range(10):
        images = pipeline(
            batch_size=6,
            generator=torch.manual_seed(seed),
            num_inference_steps=steps,
        ).images
        
        # make_grid(images).show()

        for i, image in enumerate(images):
            image.save(f'./samples/{n}_{steps}_{i}.png')


        seed += 1
