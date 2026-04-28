from PIL import Image as img
from PIL.Image import Image
import sys
from PIL import ImageDraw 
from PIL import ImageFont
from tqdm import tqdm
import numpy as np

img.MAX_IMAGE_PIXELS = 933120000

filters = range(6)

def resize(img_path, sizes, resample=img.Resampling.LANCZOS):
    im = img.open(img_path)
    with tqdm(total=len(sizes), desc=f"Resizing {img_path}") as pbar:
        for w, h in sizes:
            resized = im.resize((w, h), resample=resample)
            resized = np.array(resized).astype(np.float32)
            # if 'DEM' in img_path:
            #     resized /= resized.max()
            #     resized *= 255
            resized = img.fromarray(resized.astype(np.uint8))
            this_path = img_path.replace("21600x10800", f"{w}x{h}")
            if ".tif" in img_path:
                this_path = this_path.replace(".tif", ".png")
            elif ".jpg" in img_path:
                this_path = this_path.replace(".jpg", ".png")
            resized.save(this_path)
            pbar.update(1)

def test_resampling(img_path, sizes):
    im = img.open(img_path)
    for w, h in sizes:
        imgs = []
        crops = {
            'full': (0, 0, w, h),
            'africa': (w//512*225, h//256*75, w//512*325, h//256*175),
            'himalayas': (w//512*350, h//256*65, w//512*410, h//256*95)
        }
        
        for crop_name in crops:
            crp = crops[crop_name]
            font = ImageFont.truetype('LiberationMono-Bold.ttf', size=(crp[3]-crp[1])//10)
            for f in filters:
                
                resized = im.resize((w, h), img.Resampling(f)).crop(crp)
                draw = ImageDraw.Draw(resized)
                draw.text((0, 0), img.Resampling(f).name, 128, font=font)
                imgs.append(resized)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resize_images.py <img_path> <resampling_mode>")
        sys.exit(1)
    img_path = sys.argv[1]
    sizes = [
        (512, 256),
        (1024, 512),
        (2048, 1024),
        (4096, 2048),
        (8192, 4096),
        (16384, 8192),
    ]
    # NEAREST = 0
    # BOX = 4
    # BILINEAR = 2
    # HAMMING = 5
    # BICUBIC = 3
    # LANCZOS = 1
    resample=img.Resampling.LANCZOS if len(sys.argv) < 3 else int(sys.argv[2])
    resize(img_path, sizes, resample)
    # test_resampling(img_path, sizes)
