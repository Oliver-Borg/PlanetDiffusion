import unittest
from .sketch_gen import quantize_list, river_mask_stats, strahler_paint_river, paint_img, numpy_get_pixel_lists, dilate_paint, temperature_paint
from .map_paster import setup, gen_image, gen_dem, get_mask_config, get_tile, change_channels, gen_pair, gen_list, gen_tile_pair, get_k_val, get_combinations, _gen_image
from .utils import PlanetConfig, image_grid, continuous_to_spread, spread_to_continuous, np_rgb, total_mem, _get_tile
from .dataset import RAMDataset, _encode
from .tiler import add_tile
import numpy as np
from PIL import Image as img
from time import time
import os
from random import randint
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import fields
import random
import torch
from diffusers.pipelines import DiffusionPipeline
from torchvision.transforms import ToPILImage
from itertools import product

class TestQuantizeList(unittest.TestCase):
    def test_uniform(self, colours=8):
        pixels = [(i, (i, i)) for i in range(256)]
        pixel_lists = quantize_list(pixels, colours, True)
        self.assertEqual(len(pixel_lists), colours)
    
    def test_non_uniform(self, colours=8):
        pixels = [(i, (i, i)) for i in range(256)]
        pixel_lists = quantize_list(pixels, colours, False)
        self.assertEqual(len(pixel_lists), colours)

    def test_colours_same(self, colours=8):
        pixels = [(i, (i, i)) for i in range(256)]
        pixel_lists0 = quantize_list(pixels, colours, True)
        pixel_lists1 = quantize_list(pixels, colours, False)
        self.assertEqual(len(pixel_lists0), len(pixel_lists1))
        self.assertEqual(pixel_lists0.keys(), pixel_lists1.keys())

    def test_multiple_colours(self):
        for colours in range(2, 17):
            self.test_uniform(colours)
            self.test_non_uniform(colours)
            self.test_colours_same(colours)


class TestRiverMaskStats(unittest.TestCase):
    '''
    Tests for the river_mask_stats function.
    
    This function returns:
    {
        "river_coverage": river_coverage,
        "river_wastage": river_wastage,
        "avg_size": avg_size,
        "num_components": num_components,
        "num_rivers": num_rivers
    }
    '''
    def river_mask_stats(self, rivers: np.ndarray, river_mask: np.ndarray, 
                              river_coverage: float, river_wastage: float, 
                              avg_size: float, num_components: int, num_rivers: int):
        
        stats = river_mask_stats(rivers, river_mask)
        self.assertEqual(stats["river_coverage"], river_coverage)
        self.assertEqual(stats["river_wastage"], river_wastage)
        self.assertEqual(stats["avg_size"], avg_size)
        self.assertEqual(stats["num_components"], num_components)
        self.assertEqual(stats["num_rivers"], num_rivers)

    def test_small_river(self):
        rivers = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]) * 255
        river_mask = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=0.5, river_wastage=0.0, avg_size=1, num_components=1, num_rivers=1)    
    def test_big_river(self):
        rivers = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]) * 255
        river_mask = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=1, river_wastage=8/9, avg_size=9, num_components=1, num_rivers=1)

    def test_missed_river(self):
        rivers = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]) * 255
        river_mask = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=0, river_wastage=1, avg_size=1, num_components=1, num_rivers=1)

    def test_empty(self):
        rivers = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]) * 255
        river_mask = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=0, river_wastage=0, avg_size=0, num_components=0, num_rivers=0)

    def test_two_components(self):
        rivers = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]
        ]) * 255
        river_mask = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=0.75, river_wastage=0.5, avg_size=3, num_components=2, num_rivers=1)
    
    def test_two_rivers(self):
        rivers = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]) * 255
        river_mask = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]
        ]).astype(bool)
        self.river_mask_stats(rivers, river_mask, river_coverage=0.5, river_wastage=0.25, avg_size=4, num_components=1, num_rivers=2)


class TestColourOptions(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_colour_grid(self):
        bucketing_modes = ["none", "uniform", "local-max", "global-max"]
        colour_map = [False, True]
        river_replaces = [False, True]
        ims = []
        texts = []
        for river_replace in river_replaces:
            for bucketing_mode in bucketing_modes:
                for colour in colour_map:
                    planet_cfg = PlanetConfig(bucketing_mode=bucketing_mode, use_colour_map=colour, 
                                              river_replace=river_replace, offset=0, iters=1)
                    setup(planet_cfg)
                    im = gen_image(0, planet_cfg)
                    ims.append(img.fromarray(im))
                    texts.append(f"BM{bucketing_mode[0]} CM{str(colour)[0]} RR{str(river_replace)[0]}")
        grid = image_grid(ims, 8, 2, texts)
        for i, im0 in enumerate(ims[:-1]):
            for im1 in ims[i+1:]:
                self.assertNotEqual((np.array(im0)-np.array(im1)).sum(), 0), f"Images {i} and {i+1} are the same"
        grid.save("ColourModes.png")
        
class TestImageModes(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_image_modes(self):
        img_mode_field = PlanetConfig.__dataclass_fields__['image_mode']
        image_modes = ['sketch-to-dem', 'dem-upscaling', 'sketch-upscaling', 'dem-inpainting', 'dem-to-satellite', 'sketch-to-satellite']
        ims = []
        texts = []
        for image_mode in image_modes:
            planet_cfg = PlanetConfig(image_mode=image_mode, 
                                        offset=0, size=1, iters=1)
            setup(planet_cfg)
            pair = list(gen_tile_pair(0, 256, 512, planet_cfg))
            if image_mode == 'dem-inpainting':
                w, h, c = pair[0].shape
                pair[0][:, w//2:, 0] = 0
            pair[0] = change_channels(pair[0], 3)
            pair[1] = change_channels(pair[1], 3)
            ims.append(img.fromarray(np.concatenate(pair, axis=1)))
            texts.append(f"{image_mode}")
        grid = image_grid(ims, len(image_modes)//3, 3, texts)
        grid.save("tests/ImageModes.png")

    @unittest.skip("Too slow")
    def test_inpainting_masks(self):
        inpainting_widths = [0.25, 0.5, 0.75, 1, 0]
        iters = 11
        imgs = []
        texts = []
        for inpainting_width in inpainting_widths:
            planet_cfg = PlanetConfig(image_mode='dem-inpainting', offset=0, size=0, iters=iters,
                                      inpainting_width=inpainting_width, inpainting_mode='both',
                                      planet_seed=0, use_mask_store=True)
            setup(planet_cfg)
            planet_dataset = RAMDataset(planet_cfg, 1, 1, True)
            dataloader = DataLoader(planet_dataset, batch_size=1, shuffle=False)
            for i, item in enumerate(dataloader):
                cond_image = item.get('cond_image')
                cond_image = cond_image.numpy()[0].transpose((1,2,0))/2+0.5
                imgs.append(img.fromarray((cond_image[:, :, 0]*255).astype(np.uint8)))
                texts.append(f"IW{inpainting_width} I{i}")
        grid = image_grid(imgs, len(inpainting_widths), iters, texts)
        grid.save("InpaintingMasks.png")


class TestCollisionModes(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_collision_modes(self):
        collision_modes = ['discard', 'blend', 'translate']
        ths = [0, 50, 100]
        ims = []
        texts = []
        for collision_mode in collision_modes:
            for th in ths:
                planet_cfg = PlanetConfig(collision_mode=collision_mode, 
                                          offset=0, size=2, iters=1, 
                                          discard_threshold=th)
                setup(planet_cfg)
                im = gen_image(0 if collision_mode=='discard' else 5920225673, planet_cfg)
                sketch, dem = im[:, im.shape[1]//2:, :], im[:, :im.shape[1]//2, :]
                dem = np_rgb(dem[:, :, 0], 'terrain')
                terrain_sketch = np_rgb(sketch[:, :, 0].copy(), 'terrain')
                terrain_sketch[sketch[:,:,2] == 255] = 255
                im = np.concatenate([terrain_sketch, dem], axis=1)
                ims.append(img.fromarray(im))
                texts.append(f"CM{collision_mode[0]} TH{th}")
        grid = image_grid(ims, 3, len(ths), texts)
        grid.save("CollisionModes.png")

class TestGaussianBlur(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_gaussian_blur(self):
        blur_amounts = [0, 1, 2, 3, 4, 5]
        ims = []
        texts = []
        for blur_amount in blur_amounts:
            planet_cfg = PlanetConfig(gaussian_blur=blur_amount, offset=5, size=0, iters=100)
            setup(planet_cfg)
            im = gen_image(0, planet_cfg)
            ims.append(img.fromarray(im))
            texts.append(f"BA{blur_amount}")
        grid = image_grid(ims, 6, 1, texts)
        grid.save("GaussianBlur.png")

class TestVariableRivers(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_variable_rivers(self):
        variable_rivers = [True, False]
        river_sizes = [2, 3, 4]
        top_n_orders = [1, 2, 3, 4]
        ims = []
        texts = []
        for top_n_order in top_n_orders:
            for variable_river in variable_rivers:
                for river_size in river_sizes:
                    planet_cfg = PlanetConfig(variable_rivers=variable_river, river_size=river_size, 
                                              offset=0, size=1, iters=1, top_n_orders=top_n_order)
                    setup(planet_cfg)
                    im = gen_image(0, planet_cfg)
                    ims.append(img.fromarray(im))
                    texts.append(f"VR{str(variable_river)[0]} RS{river_size} TN{top_n_order}")
        grid = image_grid(ims, len(variable_rivers)*len(top_n_orders), len(river_sizes), texts)
        grid.save("VariableRivers.png")
    @unittest.skip("Too slow")
    def test_min_size(self):
        min_sizes = [0, 10, 20, 30, 40, 50]
        river_sizes = [2, 3, 4]
        ims = []
        texts = []
        for min_size in min_sizes:
            for river_size in river_sizes:
                planet_cfg = PlanetConfig(min_size=min_size, river_size=river_size, 
                                          offset=0, size=0, iters=1)
                setup(planet_cfg)
                im = gen_image(0, planet_cfg)
                ims.append(img.fromarray(im))
                texts.append(f"MS{min_size} RS{river_size}")
        grid = image_grid(ims, len(min_sizes), len(river_sizes), texts)
        grid.save("MinSize.png")


class TestRandomizeParameters(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_randomize_parameters(self):
        planet_cfg = PlanetConfig(planet_seed=0, offset=0, iters=1, size=0)
        setup(planet_cfg)
        planet_cfg.randomize_parameters()
        setup(planet_cfg)
        planet_cfg.randomize_parameters()
        setup(planet_cfg)
        planet_cfg.randomize_parameters()
        setup(planet_cfg)

class TestMaskTimes(unittest.TestCase):
    def setUp(self):
        self.sizes = 5
        self.extensions = ['png', 'jpeg', 'tif']
        for ext in self.extensions:
            for size in range(self.sizes):
                planet_cfg = PlanetConfig(planet_seed=0, offset=0, iters=1, size=size, image_extension=ext)
                setup(planet_cfg)
        
    @unittest.skip("Too slow")
    def test_reload_times(self):
        for ext in self.extensions:
            for size in range(self.sizes):
                planet_cfg = PlanetConfig(planet_seed=0, offset=0, iters=1, size=size, image_extension=ext)
                t1 = time()
                planet_cfg.get_mask_dicts(force=True)
                gen_image(0, planet_cfg)
                t2 = time()
                print(f"Reload: Size {size} for {ext} took {(t2-t1)*1000: .0f} ms")

    @unittest.skip("Too slow")
    def test_repaint_times(self):
        times = []
        for size in range(self.sizes):
            planet_cfg = PlanetConfig(planet_seed=0, offset=0, iters=1, size=size)
            conf = get_mask_config(0, planet_cfg)
            dem_dict, _, _, _, _ = planet_cfg.get_mask_dicts()
            t1 = time()
            dem = gen_dem(conf, dem_dict, planet_cfg)[:, :, 0]
            mask = (dem > 0).astype(np.uint8)*255
            river = strahler_paint_river(dem, mask, mask, planet_cfg)
            pixel_lists = numpy_get_pixel_lists(dem, mask, planet_cfg) 
            sketch = paint_img(dem, planet_cfg)
            to_return = np.dstack((sketch, sketch, sketch))
            if planet_cfg.river_replace:
                to_return[:, :, 2] = river
            else:
                to_return[:, :, 2][river == 255] = 255
            t2 = time()
            times.append(t2-t1)
        for size, t in enumerate(times):
            print(f"Repaint: Size {size} took {t*1000: .3f} ms")

class TestLostIslands(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_lost_islands(self):
        print(f"Size\tPixels")
        for size in range(5):
            planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=size, offset=0)
            setup(planet_cfg)
            im = gen_image(0, planet_cfg)
            im = im[:, im.shape[1]//2:, 0]
            dem = np.array(img.open(os.path.join(planet_cfg.data_dir, f"World_DEM_{planet_cfg.get_dims_str()}.png")))
            im_mask = im > 0
            dem_mask = dem > 0
            im_mask[dem_mask] = False
            im = img.fromarray(im_mask)
            types, counts = np.unique(im_mask, return_counts=True)
            print(f"{size}\t{counts[1] if len(counts) > 1 else 0}")
            im.save(f"LostIslands_{size}.png")

class TestSetup(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_size_5(self):
        planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=5, offset=0, use_mask_store=True)
        setup(planet_cfg, force=True)
        im = gen_image(124368124, planet_cfg)
        img.fromarray(im).save("TestSetup_5.png")

class TestMaskStore(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_mask_store(self):
        planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=0, offset=0, use_mask_store=False)
        setup(planet_cfg, force=True)
        im0 = gen_image(0, planet_cfg)
        planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=0, offset=0, use_mask_store=True)
        setup(planet_cfg, force=True)
        im1 = gen_image(0, planet_cfg)
        self.assertEqual((im1-im0).sum(), 0)

from torch.utils.data import DataLoader
class TestDropout(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_conditioning_dropout(self):
        iters=1000
        dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        planet_cfg = PlanetConfig(planet_seed=0, iters=iters, size=0, offset=0, use_mask_store=True)
        setup(planet_cfg)
        for dropout in dropouts:
            planet_dataset = RAMDataset(planet_cfg, 1, 2, True, dropout)
            dataloader = DataLoader(planet_dataset, batch_size=1, shuffle=False)
            empties = 0
            for i, item in enumerate(dataloader):
                cond_image = item.get('cond_image')
                cond_image = cond_image.numpy()[0].transpose((1,2,0))/2+0.5
                empty = cond_image.sum() == 0
                if empty:
                    empties += 1
            # print(f"Dropout: {dropout} had {empties/iters:0.3f} empty conditioning images")
            self.assertAlmostEqual(empties/iters, dropout, delta=0.05)


    @unittest.skip("Too slow")
    def test_river_dropout(self):
        iters=1000
        dropouts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        
        for dropout in dropouts:
            dropout_cfg = PlanetConfig(planet_seed=0, iters=iters, size=0, 
                                        offset=0, use_mask_store=True, river_dropout=dropout)
            planet_cfg = PlanetConfig(planet_seed=0, iters=iters, size=0,
                                        offset=0, use_mask_store=True, river_dropout=0)
            setup(dropout_cfg)
            setup(planet_cfg)
            dropout_dataset = RAMDataset(dropout_cfg, 1, 2, True)
            planet_dataset = RAMDataset(planet_cfg, 1, 2, True)
            dropout_dataloader = DataLoader(dropout_dataset, batch_size=1, shuffle=False)
            planet_dataloader = DataLoader(planet_dataset, batch_size=1, shuffle=False)
            empties = 0
            for i, item in enumerate(dropout_dataloader):
                cond_image = item.get('cond_image')
                cond_image = cond_image.numpy()[0].transpose((1,2,0))/2+0.5
                empty = cond_image[:, :, 1].sum() == 0
                if empty:
                    empties += 1
            total = 0
            for i, item in enumerate(planet_dataloader):
                cond_image = item.get('cond_image')
                cond_image = cond_image.numpy()[0].transpose((1,2,0))/2+0.5
                empty = cond_image[:, :, 1].sum() == 0
                if empty:
                    total += 1
            # print(f"Dropout: {dropout} had {(empties-total)/(iters-total):0.3f} empty river channels")
            self.assertAlmostEqual((empties-total)/(iters-total), dropout, delta=0.05)

class TestGetTile(unittest.TestCase):
    def test_in_bounds(self):
        test_array = np.ones((256, 256))
        tile = get_tile(test_array, 0, 0, 256)
        self.assertEqual(tile.shape, (256, 256))
    
    def test_irregular(self):
        test_array = np.ones((256, 512))
        tile = get_tile(test_array, 256, 0, 256)
        self.assertEqual(tile.shape, (256, 256))

    def test_different_width(self):
        test_array = np.ones((256, 512))
        tile = get_tile(test_array, 256, 0, 128)
        self.assertEqual(tile.shape, (128, 128))

    def test_out_of_bounds(self):
        test_array = np.ones((256, 256))
        tile = get_tile(test_array, 256, 0, 256, edge_pad=False)
        self.assertEqual(tile.shape, (256, 256))
        self.assertEqual(tile.sum(), 0)

        tile = get_tile(test_array, -128, 0, 256, edge_pad=False)
        self.assertEqual(tile.shape, (256, 256))
        self.assertEqual(tile.sum(), 128*256)

class TestParentDem(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_image_modes(self):
        images_modes = ['sketch-to-dem', 'sketch-inpainting', 'dem-inpainting', 'sketch-upscaling', 'sat']
        use_parent_dems = [True, False]
        images = []
        text_grid = []
        for use_parent_dem in use_parent_dems:
            for image_mode in images_modes:
                planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=1, offset=0,
                                            use_mask_store=True, image_mode=image_mode, 
                                            use_parent_dem=use_parent_dem)
                setup(planet_cfg)
                im = gen_image(0, planet_cfg)
                images.append(img.fromarray(im))
                text_grid.append(f"PD{str(use_parent_dem)} {image_mode}")
        grid = image_grid(images, len(use_parent_dems), len(images_modes), text_grid)
        grid.save("TestParentDemImageModes.png")

    @unittest.skip("Too slow")
    def test_padding(self):
        pads = [0, 32, 64, 256]
        images_modes = ['sketch-to-dem', 'sketch-inpainting', 'dem-inpainting', 'sketch-upscaling', 'sat']
        out_channels = [1, 2, 1, 2]
        in_channels = [3, 3, 2, 3]
        use_parent_dems = [True, False]
        images = []
        text_grid = []
        for use_parent_dem in use_parent_dems:
            for pad in pads:
                for image_mode, o, i in zip(images_modes, out_channels, in_channels):
                    planet_cfg = PlanetConfig(planet_seed=42, iters=1, size=1, offset=0,
                                                use_mask_store=True, image_mode=image_mode, 
                                                use_parent_dem=use_parent_dem, parent_dem_padding=pad)
                    setup(planet_cfg)
                    if not use_parent_dem:
                        i-=1
                    dataset = RAMDataset(planet_cfg, o, i, True)
                    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                    item = next(iter(dataloader))
                    cond_image = item['cond_image'].numpy()[0].transpose((1,2,0))/2+0.5
                    target_image = item['target_image'].numpy()[0].transpose((1,2,0))/2+0.5
                    cond_image = (cond_image*255).astype(np.uint8)
                    target_image = (target_image*255).astype(np.uint8)
                    if 'upscaling' in image_mode:
                        self.assertEqual(cond_image.shape[0], 128)
                        self.assertEqual(cond_image.shape[1], 128)

                    cond_image = change_channels(cond_image, 3)
                    target_image = change_channels(target_image, 3)
                    
                    images.append(img.fromarray(cond_image))
                    images.append(img.fromarray(target_image))
                    text_grid.append(f"{image_mode}")
                    text_grid.append(f"PD{str(use_parent_dem)} P{pad}")
        grid = image_grid(images, len(use_parent_dems)*len(pads), 2*len(images_modes), text_grid)
        grid.save("TestParentDem.png")   

class TestTiler(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_tiler_modes(self):
        merge_modes = [1]
        overlap_widths = [0, 16, 32, 64]
        repetitions = 10
        size = 3
        images = []
        text_grid = []
        planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=size, offset=0,
                                                    use_mask_store=True)
        setup(planet_cfg)
        _, original_dem = gen_pair(0, planet_cfg)
        with tqdm(total=len(merge_modes)*len(overlap_widths)*repetitions) as pbar:
            for repetition in range(repetitions):
                source_y = randint(0, 2**size-1)
                source_x = randint(0, 2**(size+1)-1)
                dest_y = randint(0, 2**(size)-1)
                dest_x = randint(0, 2**(size+1)-1)
                for merge_mode in merge_modes:
                    for overlap_width in overlap_widths:
                        
                        height, width = 256 * 2 ** (size) - (2 ** (size) - 1)*overlap_width, 256 * 2 ** (size+1) - (2 ** (size))*overlap_width
                        dem = original_dem.copy()
                        tile = get_tile(dem, 256*source_y, 256*source_x)[:, :, 0].copy().astype(np.float32)/255
                        dem = cv2.resize(dem, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)/255
                        dem = add_tile(dem, tile, dest_y, dest_x, size, merge_mode, overlap_width)
                        images.append(img.fromarray((dem*255).astype(np.uint8)))
                        text_grid.append(f"MM{str(merge_mode)} OW{overlap_width} Y{dest_y} X{dest_x}")
                        pbar.update(1)
        grid = image_grid(images, len(merge_modes)*repetitions, len(overlap_widths), text_grid)
        grid.save("TestTilerModes.png")

class TestOffsets(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_offsets(self):
        for size in range(6):  
            default_cfg = PlanetConfig(planet_seed=0, iters=1, size=size, offset=5,
                                                    use_mask_store=True) 
            setup(default_cfg)
            _, default_dem = gen_pair(0, default_cfg)
            for offset in range(6-size):
                cfg = PlanetConfig(planet_seed=0, iters=1, size=size, offset=offset,
                                                    use_mask_store=True) 
                setup(cfg)
                _, dem = gen_pair(0, cfg)

                # Calculate mse
                mae = np.mean(np.abs(default_dem.astype(np.float32) - dem.astype(np.float32)))
                max_error = np.max(np.abs(default_dem.astype(np.float32) - dem.astype(np.float32)))
                print(f"Size: {size} Offset: {offset} MAE: {mae} MAX: {max_error}")

class TestDataLoader(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_dataloader(self):
        batch_size = 8
        target_image_channels = 1
        cond_image_channels = 1
        conditioning_dropout = 0.1
        num_worksers = 2


        iters = 100000
        image_mode = 'normal-noise'
        size = 0
        use_mask_store = True
        planet_seed = 0
        planet_cfg = PlanetConfig(
            planet_seed=planet_seed, iters=iters, size=size,
            use_mask_store=use_mask_store, image_mode=image_mode)
        
        setup(planet_cfg)
        dataset = RAMDataset(planet_cfg, target_image_channels, cond_image_channels, True, conditioning_dropout)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worksers)
        with tqdm(total=iters) as pbar:
            for i, item in enumerate(dataloader):
                pbar.update(batch_size)

class TestDilate(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_dilate(self):
        erode_sizes = [1, 3]
        dilate_sizes = [1, 3, 5]
        erode_iters = [0, 1]
        dilate_iters = [0, 1]
        dilate_firsts = [True, False]
        gaussian_blurs = [0, 1, 10]
        sketch_modes = ['brush', 'dilate']
        images = []
        texts = []
        dem = np.array(img.open("data/World_DEM_512x256.png"))
        for gaussian_blur in gaussian_blurs:
                for sketch_mode in sketch_modes:
                    cfg = PlanetConfig(planet_seed=0, iters=1, size=0, offset=5,
                                        use_mask_store=True, sketch_mode=sketch_mode, 
                                        bucketing_mode='global-max', gaussian_blur=gaussian_blur)
                    if sketch_mode == 'dilate':
                        for erode_size in erode_sizes:
                            for dilate_size in dilate_sizes:
                                for erode_iter in erode_iters:
                                    for dilate_iter in dilate_iters:
                                        for dilate_first in dilate_firsts:
                                            cfg.erode_size = erode_size
                                            cfg.dilate_size = dilate_size
                                            cfg.erode_iters = erode_iter
                                            cfg.dilate_iters = dilate_iter
                                            cfg.dilate_first = True
                                            sketch = dilate_paint(dem, cfg)
                                            images.append(img.fromarray(sketch))
                                            texts.append(f"ES: {erode_size} EI: {erode_iter} DS: {dilate_size} DI: {dilate_iter} GB: {gaussian_blur} DF: {dilate_first}")
                    else:
                        sketch = paint_img(dem, cfg)
                        images.append(img.fromarray(sketch))
                        texts.append(f"GB: {gaussian_blur} SK: {sketch_mode}")
        grid = image_grid(images, len(gaussian_blurs), 1+len(erode_sizes)*len(erode_iters)*len(dilate_sizes)*len(dilate_iters)*len(dilate_firsts), texts)
        grid.save("TestDilate.png")
        

class TestDataLoader(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_global_max_dataloader(self):
        for size in range(1):
            planet_cfg = PlanetConfig(planet_seed=0, iters=10000, size=size, offset=5,
                                    bucketing_mode='global-max', use_mask_store=True)
            setup(planet_cfg)
            dataset = RAMDataset(planet_cfg, 1, 1, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            mx = -1
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    mx = max(mx, dem.max())
                    pbar.update(1)
            self.assertEqual(mx, 1)
    # @unittest.skip("Too slow")
    def test_dataloader_images(self):
        rows = 100
        cols = 1
        sizes = [4]
        downscale_offsets = [4]
        use_quad_datas = [False, True]
        for size, downscale_offset, use_quad_data in product(sizes, downscale_offsets, use_quad_datas):
            planet_cfg = PlanetConfig(
                iters=rows*cols, size=size, image_mode='sat', 
                downscale_offset=downscale_offset, 
                extra_rotations=True, rotation_angle=1, preserve_edges=True, 
                dilate_size=1, offset=0, dilate_first=True, dilate_iters=1, 
                erode_iters=0, use_surrounds=True, inpainting_width=0.45,
                context_dropout=0.0, use_quad_data=use_quad_data, inpainting_channels=0,
                use_rivers=True
            )
            # setup(planet_cfg, force=False)
            dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            images = []
            texts = []
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader):
                    if i >= planet_cfg.iters:
                        break
                    dem = item['target_image']
                    sketch = item['cond_image']
                    output = item['target_image']
                    metadata = item['metadata']

                    embedding = metadata['embedding'][0].numpy().astype(int)
                    dem_encoding = np.argwhere(embedding[:planet_cfg.colours+1] == 1)[0][0]
                    land_encoding = np.argwhere(embedding[planet_cfg.colours+1:planet_cfg.colours+1+planet_cfg.landcover_classes+1] == 1)[0][0]
                    temp_encoding = np.argwhere(embedding[planet_cfg.colours+1+planet_cfg.landcover_classes+1:planet_cfg.colours+1+planet_cfg.landcover_classes+1+planet_cfg.temp_classes+1] == 1)[0][0]
                    feature_encoding_start = planet_cfg.colours + 1 + planet_cfg.landcover_classes + 1 + planet_cfg.temp_classes + 1
                    feature_encoding = embedding[feature_encoding_start:]
                    _coastline, _mountains, _snow_cover, _rivers, _islands, _icebergs, _lakes = feature_encoding[:7]
                    grid = planet_cfg.output_display(sketch, dem, output)
                    to_display = (grid.cpu().numpy().transpose((1, 2, 0))*127.5+127.5).astype(np.uint8)
                    # to_display = np_rgb(to_display, 'terrain')
                    images.append(img.fromarray(to_display))
                    texts.append(f"coastline: {_coastline} mountains: {_mountains} snow_cover: {_snow_cover} rivers: {_rivers} islands: {_islands} icebergs: {_icebergs} lakes: {_lakes} dem: {dem_encoding} land: {land_encoding} temp: {temp_encoding}")
                    pbar.update(1)
            grid = image_grid(images, rows, cols, texts=texts)
            grid.save(f"tests/TestDataLoader{size}_{downscale_offset}_{use_quad_data}.png")

    @unittest.skip("Too slow")
    def test_uniqueness(self):
        iters = 10000
        planet_cfg = PlanetConfig(image_mode='land', size=5)
        dataset = RAMDataset(planet_cfg)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        precision = 5
        count = 0
        total_ks = 0
        means = [[] for _ in range(256*10**precision)]
        with tqdm(total=len(dataloader)) as pbar:
            for i, item in enumerate(dataloader): 
                dem = item['target_image']
                sketch = item['cond_image']
                metadata = item['metadata']
                k = int(metadata['k'][0])
                y = int(metadata['tile_y'][0])
                x = int(metadata['tile_x'][0])
                dem = dem.squeeze().numpy()*127.5+127.5
                index = int(dem.mean()*10**precision)
                if means[index] == []:
                    count += 1 
                if (k, y, x) not in means[index]:
                    means[index].append((k, y, x))
                    total_ks += 1
                pbar.update(1)
        dupes = list(filter(lambda x: len(x) > 1, means))
        for dupe_list in dupes:
            # images = []
            arrays = []
            for k, y, x in dupe_list:
                cfg = get_mask_config(k, planet_cfg)
                dem = _gen_image({}, cfg, 'dem', planet_cfg, 'max', y, x)
                # images.append(img.fromarray(dem))
                arrays.append(dem)
            # grid = image_grid(images, 1, len(dupe_list))
            extra_uniques = len(dupe_list) - 1 # Subtract one for the first one
            for i, dem in enumerate(arrays[:-1]):
                for j, dem2 in enumerate(arrays[i+1:]):
                    if np.array_equal(dem, dem2):
                        extra_uniques -= 1
                        break
            count += extra_uniques
                        

        unique = count / total_ks
        print(f"Unique: {count} / {total_ks} = {100*unique:.2f}%")

    @unittest.skip("Too slow")
    def test_multiple_cfgs(self):
        rows = 10
        cols = 10
        planet_seed = 0
        planet_cfg = PlanetConfig(iters=rows*cols, size=1, offset=0, force_gen_list=True,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    extra_rotations=True, planet_seed=planet_seed, randomize_steps=cols,
                                    on_fly_conditioning=True, on_fly_save=True)
        setup(planet_cfg)
        num_cfgs = 10
        # cfgs = [replace(planet_cfg, planet_seed=planet_seed+i+1 if planet_seed is not None else None) for i in range(num_cfgs)]
        dataset = RAMDataset(planet_cfg, 1, 1, True, 0, num_configurations=num_cfgs)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        images = []
        stacked = 0
        with tqdm(total=len(dataloader)) as pbar:
            for i, item in enumerate(dataloader): 
                dem = item['target_image']
                sketch = item['cond_image']
                metadata = item['metadata']
                if metadata['stacked'][0]:
                    stacked += 1
                dem = dem.squeeze().numpy()*127.5+127.5
                sketch = sketch.squeeze().numpy()*127.5+127.5
                to_display = np.concatenate([sketch, dem], axis=1).astype(np.uint8)
                to_display = np_rgb(to_display, 'terrain')
                images.append(img.fromarray(to_display))
                pbar.update(1)
        print(f"Stacked: {stacked}/{len(dataloader)}")
        grid = image_grid(images, rows, cols)
        grid.save("TestMultipleCfgs.png")

    @unittest.skip("Too slow")
    def test_gen_lists(self):
        rows = 10
        cols = 10
        images = []
        for gen_lists in [True, False]:
            t1 = time()
            planet_cfg = PlanetConfig(iters=rows*cols, size=0, force_gen_list=False,
                                        bucketing_mode='global-max', extra_rotations=True, 
                                        planet_seed=0, gen_lists=gen_lists, randomize_steps=rows*cols)
            setup(planet_cfg)
            dataset = RAMDataset(planet_cfg, 1, 1, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    sketch = item['cond_image']
                    dem = dem.squeeze().numpy()*127.5+127.5
                    sketch = sketch.squeeze().numpy()*127.5+127.5
                    to_display = np.concatenate([sketch, dem], axis=1).astype(np.uint8)
                    to_display = np_rgb(to_display, 'terrain')
                    images.append(img.fromarray(to_display))
                    pbar.update(1)
            t2 = time()
            print(f"Gen Lists: {gen_lists} took {t2-t1: .3f} s")
        grid = image_grid(images, rows, cols*2)
        grid.save("TestGenLists.png")

    @unittest.skip("Too slow")
    def test_sea_level(self):
        cols = 5
        default_cfg = PlanetConfig(num_configurations=1)
        sea_levels = list(set(default_cfg.values['sea_level']))
        rows = len(sea_levels)
        images = []
        texts = []
        row_labels = []
        col_labels = [f"SLR: {sea_level_rescale}" for sea_level_rescale in [True, False]]*cols
        for row in range(rows):
            row_labels.append(f"SL: {sea_levels[row]}")
            for sea_level_rescale in [True, False]:
                sea_level = sea_levels[row]
                planet_cfg = PlanetConfig(iters=cols, size=0, offset=5, force_gen_list=True,
                                            bucketing_mode='global-max', use_mask_store=True,
                                            extra_rotations=True, sea_level=sea_level, 
                                            sea_level_rescale=sea_level_rescale,
                                            on_fly_conditioning=True, planet_seed=0,
                                            on_fly_save=True, num_configurations=1)
                setup(planet_cfg)
                dataset = RAMDataset(planet_cfg, 1, 1, True, 0)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                with tqdm(total=len(dataloader)) as pbar:
                    for i, item in enumerate(dataloader): 
                        dem = item['target_image']
                        sketch = item['cond_image']
                        dem = dem.squeeze().numpy()*127.5+127.5
                        sketch = sketch.squeeze().numpy()*127.5+127.5
                        to_display = np.concatenate([sketch, dem], axis=1).astype(np.uint8)
                        to_display = np_rgb(to_display, 'terrain')
                        images.append(img.fromarray(to_display))
                        pbar.update(1)
        grid = image_grid(images, rows, cols*2, col_labels=col_labels, row_labels=row_labels)
        grid.save("TestSeaLevel.png")
    
    @unittest.skip("Too slow")
    def test_sea_level_figure(self):
        sea_levels = [0, 10, 1, 15, 2, 20, 3, 30, 4, 50]
        images = []
        texts = []
        for sea_level in sea_levels:
            planet_cfg = PlanetConfig(sea_level=sea_level)
            sketch0, dem0 = gen_tile_pair(0, 0, 0, planet_cfg)
            sketch1, dem1 = gen_tile_pair(0, 0, 256, planet_cfg)
            sketch = np.concatenate([sketch0, sketch1], axis=1)
            dem = np.concatenate([dem0, dem1], axis=1)
            pair = np.concatenate([sketch[:, :, 0], dem[:, :, 0]], axis=1)
            images.append(img.fromarray(cv2.resize(np_rgb(pair), (0, 0), fx=4, fy=4)))
            texts.append(f"{sea_level}")
        grid = image_grid(images, rows=len(sea_levels)//2, cols=2, texts=texts)
        grid.save('tests/sea_level.png')



    @unittest.skip("Too slow")
    def test_randomize_steps(self):
        offset = 0
        iters = 1000
        num_workers = 4
        randomize_steps = iters//2
        use_parent_dem = False
        cond_channels = 2
        for size in range(0, 1):
            t1 = time()
            planet_cfg = PlanetConfig(planet_seed=0, iters=iters, size=size, offset=offset,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=use_parent_dem)
            setup(planet_cfg)
            dataset = RAMDataset(planet_cfg, 1, cond_channels, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    pbar.update(1)
            t2 = time()
            
            planet_cfg = PlanetConfig(planet_seed=None, iters=iters, size=size, offset=offset,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=use_parent_dem, randomize_steps=randomize_steps)
            setup(planet_cfg, force=True)
            dataset = RAMDataset(planet_cfg, 1, cond_channels, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    pbar.update(1)

            t3 = time()
            planet_cfg = PlanetConfig(extra_rotations=True, planet_seed=0, iters=iters, size=size, offset=offset,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=use_parent_dem, randomize_steps=randomize_steps)

            dataset = RAMDataset(planet_cfg, 1, cond_channels, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    pbar.update(1)

            t4 = time()

            planet_cfg = PlanetConfig(planet_seed=None, iters=iters, size=size, offset=offset,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=use_parent_dem, randomize_steps=randomize_steps,
                                    on_fly_conditioning=True)
            
            dataset = RAMDataset(planet_cfg, 1, cond_channels, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    dem = item['target_image']
                    pbar.update(1)
                
            t5 = time()

            print(f"Size: {size} No Randomize: {t2-t1:0.2f} Randomize (force): {t3-t2:0.2f} Randomize: {t4-t3:0.2f} On Fly (Conditioning): {t5-t4:0.2f}")
    
    @unittest.skip("Too slow")
    def test_random_sketches(self):
        rows = 3
        cols = 5
        offset = 0
        num_workers = 0
        randomize_steps = 1
        use_parent_dem = False
        cond_channels = 1
        imgs = []
        row_labels = [f'Zoom: {size}' for size in range(rows)]
        with tqdm(total=rows) as pbar:
            for size in range(rows):
                planet_cfg = PlanetConfig(planet_seed=None, iters=cols, size=size, offset=offset,
                                            bucketing_mode='global-max', use_mask_store=True,
                                            use_parent_dem=use_parent_dem, randomize_steps=randomize_steps,
                                            on_fly_conditioning=True, num_configurations=1)
                setup(planet_cfg)
                dataset = RAMDataset(planet_cfg, 1, cond_channels, True, 0)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            
            
                for i, item in enumerate(dataloader): 
                    sketch = item['cond_image']
                    dem = item['target_image']
                    sketch = (sketch.squeeze(0).numpy().transpose(1, 2, 0)*127.5+127.5).astype(np.uint8)
                    dem = (dem.squeeze(0).numpy().transpose(1, 2, 0)*127.5+127.5).astype(np.uint8)
                    imgs.append(img.fromarray(np_rgb(np.concatenate([sketch[:, :, 0], dem[:, :, 0]], axis=1))))
                pbar.update(1)

        grid = image_grid(imgs, rows, cols, row_labels=row_labels, padding_colour=255, size_multiplier=4)
        grid.save("tests/TestRandomSketches.png")
        print("Saved")

    @unittest.skip("Too slow")
    def test_add_size_to_colours(self):
        imgs = []
        iters = 1000
        display = 10
        sizes = [0, 1, 2, 3, 4]
        for size in sizes:
            planet_cfg = PlanetConfig(planet_seed=0, iters=iters, size=size, offset=0,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=False, add_size_to_colours=True, 
                                    on_fly_conditioning=True, brush_size=5, 
                                    sketch_mode='dilate', erode_iters=3, dilate_size=5,
                                    dilate_iters=2, dilate_first=True)
            setup(planet_cfg)
            dataset = RAMDataset(planet_cfg, 3, 3, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            with tqdm(total=display) as pbar:
                for i, item in enumerate(dataloader):
                    if i >= display:
                        break 
                    sketch = item['cond_image']
                    dem = item['target_image']
                    sketch = (sketch.squeeze().numpy().transpose(1, 2, 0)*127.5+127.5).astype(np.uint8)
                    dem = (dem.squeeze().numpy().transpose(1, 2, 0)*127.5+127.5).astype(np.uint8)
                    imgs.append(img.fromarray(np.concatenate([sketch, dem], axis=1)))
                    pbar.update(1)
        grid = image_grid(imgs, len(sizes), display)
        grid.save("TestAddSizeToColours.png")
        
class TestMaxPixel(unittest.TestCase):
    def test_max_pixel(self):
        for size in range(6):
            planet_cfg = PlanetConfig(size=size, planet_seed=size, bucketing_mode='global-max')
            print(planet_cfg.dem_max_pixel)

class GenListTests(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_distribution(self):
        planet_cfg = PlanetConfig(extra_rotations=True, planet_seed=0, iters=100000)
        setup(planet_cfg)
        to_gen = gen_list(planet_cfg=planet_cfg, tiles=True)
        operations = planet_cfg.operations
        counts = {}
        big_ops = ['mask_refs', 'mask_angs', 'mask_rems']
        for big_op in big_ops:
            counts[big_op] = {}
        counts['extras'] = {}
        extra_ops = ['hflip', 'vflip', 'spin']
        for k, y, x in to_gen:
            cfg = get_mask_config(k, planet_cfg)
            for op in big_ops:
                for i, val in enumerate(cfg[op]):
                    if cfg['mask_rems'][i] == True and op != 'mask_rems':
                        continue
                    key = f"{i}_{val}"
                    if key not in counts[op]:
                        counts[op][key] = 0
                    counts[op][key] += 1
            for op in extra_ops:
                key = f"{op}_{cfg[op]}"
                if key not in counts['extras']:
                    counts['extras'][key] = 0
                counts['extras'][key] += 1
        print(counts)
        # Sort counts dictionary by key
        for key in counts:
            counts[key] = {k: v for k, v in sorted(counts[key].items(), key=lambda item: item[0])}

        # Plot counts dictionary
        #'key': value
        # Add each plot as a subplot
        fig, axs = plt.subplots(len(counts), 1, figsize=(10, 10))
        for i, key in enumerate(counts):
            axs[i].bar(counts[key].keys(), counts[key].values())
            axs[i].set_title(key)
            axs[i].set_xticklabels(counts[key].keys(), rotation=90)
        plt.tight_layout()
        plt.show()

class TestSatellite(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_satellite_dists(self):
        cfg = PlanetConfig(planet_seed=0, iters=1, size=0, offset=5, image_mode='sketch-to-satellite',
                           on_fly_conditioning=True, extra_rotations=False, gaussian_blur=10, blur_size=7,
                           brush_size=2, erode_size=2, erode_iters=2)
        setup(cfg)
        # sketch0, sat0 = gen_tile_pair(0, 0, 0, cfg)
        # sketch1, sat1 = gen_tile_pair(0, 0, 256, cfg)
        # sketch = np.concatenate([sketch0, sketch1], axis=1)
        # sat = np.concatenate([sat0, sat1], axis=1)
        sketch, sat = gen_pair(0, cfg)
        im = np.concatenate([sat, sketch], axis=0)
        palette = img.open("Palette.jpg")
        palette = np.array(palette)
        height, width = im.shape[:2]
        palette_height, palette_width = palette.shape[:2]
        fs = height/palette_height
        palette = cv2.resize(palette, (0, 0), fx=fs, fy=fs)
        im = np.concatenate([im, palette], axis=1)
        img.fromarray(im).save("tests/TestSatelliteSketch.png")


class TestSketchParams(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_brush_sketch_params(self):
        sketch_params = {
            'brush_size': [2, 3, 4, 5],
            'sketch_colours': [3, 4, 5, 6],
            'bucketing_mode': ['global-max', 'none', 'local-max', 'uniform'],
            'gaussian_blur': [0, 1, 5, 10],
            'blur_size': [1, 3, 5, 7],
        }
        names = {
            'brush_size': 'Brush Size',
            'sketch_colours': 'Sketch Colours',
            'bucketing_mode': 'Bucketing Mode',
            'gaussian_blur': 'Gaussian Blur',
            'blur_size': 'Blur Size',
        }
        os.makedirs("tests/test_sketch_params", exist_ok=True)
        images = []
        texts = []
        for k in sketch_params:
            for v in sketch_params[k]:
                planet_cfg = PlanetConfig(**{k: v}, on_fly_conditioning=True)
                setup(planet_cfg)
                sketch, dem = gen_tile_pair(0, 0, 0, planet_cfg)
                sketch = np.concatenate([sketch, gen_tile_pair(0, 0, 256, planet_cfg)[0]], axis=1)
                dem = np.concatenate([dem, gen_tile_pair(0, 0, 256, planet_cfg)[1]], axis=1)
                # Make sketch and dem 4x bigger to have better quality text
                sketch = cv2.resize(sketch, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                dem = cv2.resize(dem, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                images.append(img.fromarray(sketch[:, :, 0]))
                texts.append(f"{v}")
            images.append(img.fromarray(dem))
            texts.append("Real")
        row_labels = list(names.values())
        # col_labels = [str(i) for i in range(len(sketch_params))]
        grid = image_grid(images, len(sketch_params), len(sketch_params), texts=texts, 
                          row_labels=row_labels, padding=10, padding_colour=128)
        grid.save("tests/test_sketch_params/brush_sketch_params.png")
    
    @unittest.skip("Too slow")
    def test_dilate_sketch_params(self):
        sketch_params = {
            'dilate_size': [2, 3, 4, 5],
            'erode_size': [1, 2, 3, 4],
            'dilate_iters': [0, 1, 2, 3],
            'erode_iters': [0, 1, 2, 3],
            'dilate_first': [True, False],
        }
        names = {
            'dilate_size': 'Dilate Size',
            'erode_size': 'Erode Size',
            'dilate_iters': 'Dilate Iterations',
            'erode_iters': 'Erode Iterations',
            'dilate_first': 'Dilate First',
        }
        os.makedirs("tests/test_sketch_params", exist_ok=True)
        images = []
        texts = []
        for k in sketch_params:
            for v in sketch_params[k]:
                planet_cfg = PlanetConfig(**{k: v, 'dilate_first': True if k == 'dilate_first' and v else False}, 
                                          on_fly_conditioning=True, sketch_mode='dilate')
                setup(planet_cfg)
                sketch, dem = gen_tile_pair(0, 0, 0, planet_cfg)
                sketch = np.concatenate([sketch, gen_tile_pair(0, 0, 256, planet_cfg)[0]], axis=1)
                dem = np.concatenate([dem, gen_tile_pair(0, 0, 256, planet_cfg)[1]], axis=1)
                # Make sketch and dem 4x bigger to have better quality text
                sketch = cv2.resize(sketch, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                dem = cv2.resize(dem, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                images.append(img.fromarray(sketch[:, :, 0]))
                texts.append(f"{v}")
            images.append(img.fromarray(dem))
            texts.append("Real")
        row_labels = list(names.values())
        # col_labels = [str(i) for i in range(len(sketch_params))]
        grid = image_grid(images, len(sketch_params), len(sketch_params), texts=texts, 
                          row_labels=row_labels, padding=10, padding_colour=128)
        grid.save("tests/test_sketch_params/dilate_sketch_params.png")
    
    @unittest.skip("Too slow")
    def test_river_params(self):
        river_params = {
            'min_size': [0, 25, 50, 75],
            'threshold': [0, 50, 100, 150],
            'top_n_orders': [0, 1, 2, 3],
            'river_size': [0, 1, 2, 3],
            'gaussian_blur': [0, 1, 5, 10],
            'blur_size': [1, 3, 5, 7],
            'variable_rivers': [True, False],
        }
        names = {
            'min_size': 'Min Size',
            'threshold': 'Threshold',
            'top_n_orders': 'Top N Orders',
            'river_size': 'River Size',
            'gaussian_blur': 'Gaussian Blur',
            'blur_size': 'Blur Size',
            'variable_rivers': 'Variable Rivers',
        }
        os.makedirs("tests/test_sketch_params", exist_ok=True)
        images = []
        texts = []
        for k in river_params:
            for v in river_params[k]:
                planet_cfg = PlanetConfig(**{'top_n_orders': 3, k: v}, on_fly_conditioning=True, size=3, offset=0, extra_rotations=False)
                setup(planet_cfg)
                sketch, dem = gen_tile_pair(0, 2*256, 11*256, planet_cfg)
                sketch = np.concatenate([sketch, gen_tile_pair(0, 2*256, 12*256, planet_cfg)[0]], axis=1)
                dem = np.concatenate([dem, gen_tile_pair(0, 2*256, 12*256, planet_cfg)[1]], axis=1)
                river_mask = sketch[:, :, 1] == 255
                dem = np.dstack([dem]*3)
                dem[river_mask] = [0, 0, 255]
                # Make 4x bigger to have better quality text
                dem = cv2.resize(dem, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
                images.append(img.fromarray(dem))
                texts.append(f"{v}")

        row_labels = list(names.values())
        grid = image_grid(images, len(river_params), 4, texts=texts, 
                          row_labels=row_labels, padding=10, padding_colour=128)
        grid.save("tests/test_sketch_params/river_params.png")

class TestMemory(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_memory(self):
        planet_cfg = PlanetConfig(planet_seed=0, iters=45, size=2, offset=5,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=False, randomize_steps=1)
        setup(planet_cfg)
        dataset = RAMDataset(planet_cfg, 1, 1, True, 0)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        memory = []
        with tqdm(total=len(dataloader)) as pbar:
            for i, item in enumerate(dataloader): 
                dem = item['target_image']
                memory.append(total_mem())
                pbar.update(1)

        plt.plot(memory)
        plt.show()

class TestExtraRotations(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_extra_rotations(self):
        planet_cfg = PlanetConfig(planet_seed=0, iters=1, size=0, offset=5,
                                    bucketing_mode='global-max', use_mask_store=True,
                                    use_parent_dem=False, randomize_steps=1, extra_rotations=True)
        setup(planet_cfg)
        mask_angs = planet_cfg.operations['ang']
        for mask, angs in enumerate(mask_angs):
            last_dem = None
            for ang in angs:
                mask_name = f'{mask}_False_{ang}.png'
                dem = planet_cfg.get_mask(mask_name, 'dem')
                if last_dem is not None:
                    mse = np.mean(np.abs((dem-last_dem)))
                    print(f"Mask: {mask} Ang: {ang} MSE: {mse}")
                last_dem = dem

class TestColourDropout(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_colour_dropout(self):
        iters = 10
        colour_dropouts = [0.0, 0.1, 0.2, 0.5, 1.0]
        images = []
        row_labels = [f"CD: {colour_dropout}" for colour_dropout in colour_dropouts]
        for colour_dropout in colour_dropouts:
            planet_cfg = PlanetConfig(colour_dropout=colour_dropout, size=1, 
                                      extra_rotations=False, iters=iters, planet_seed=0)
            setup(planet_cfg)
            dataset = RAMDataset(planet_cfg, 1, 1, True, 0)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            with tqdm(total=len(dataloader)) as pbar:
                for i, item in enumerate(dataloader): 
                    sketch = item['cond_image']
                    sketch = sketch.squeeze().numpy()*127.5+127.5
                    to_display = sketch.astype(np.uint8)
                    images.append(img.fromarray(to_display))
                    pbar.update(1)
        grid = image_grid(images, len(colour_dropouts), iters, row_labels=row_labels)
        grid.save("tests/TestColourDropout.png")


class ImConfTests(unittest.TestCase):
    # @unittest.skip("Too slow")
    def test_im_conf(self):
        planet_cfg = PlanetConfig()
        max_k = get_combinations(planet_cfg=planet_cfg)
        cfg = get_mask_config(max_k-1, planet_cfg)
        for i in range(100):
            k = random.randint(0, max_k-1)
            cfg = get_mask_config(k, planet_cfg)
            j = get_k_val(cfg, planet_cfg)
            new_cfg = get_mask_config(j, planet_cfg)
            self.assertEqual(new_cfg, cfg)

class ImageAlignTest(unittest.TestCase):
    @unittest.skip("Too slow")
    def test_sat_dem_align(self):
        rows = 30
        cols = 1
        planet_cfg = PlanetConfig(iters=rows*cols, size=5, image_mode='dem-to-satellite', 
                                          downscale_offset=5, 
                                          extra_rotations=True, rotation_angle=1, preserve_edges=True, 
                                          dilate_iters=0, erode_iters=0, offset=0, inpainting_channels=0)
        setup(planet_cfg, force=False)
        dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        images = []
        with tqdm(total=len(dataloader)) as pbar:
            for i, item in enumerate(dataloader): 
                target = item['target_image']
                cond = item['cond_image']
                output = item['target_image']
                grid = planet_cfg.output_display(cond, target, output)
                to_display = (grid.cpu().numpy().transpose((1, 2, 0))*127.5+127.5).astype(np.uint8)
                dem = to_display[:, :256, :]
                sat = to_display[:, 256:512, :]
                sat_mask = (sat[:, :, 0] > 0) & (sat[:, :, 1] > 0) & (sat[:, :, 2] > 0)
                dem_mask = dem[:, :, 0] > 0
                extra = np.dstack([np.zeros_like(sat_mask), sat_mask, dem_mask]).astype(np.uint8)*255
                to_display = np.concatenate([to_display, extra], axis=1)
                images.append(img.fromarray(to_display))
                pbar.update(1)
        grid = image_grid(images, rows, cols)
        grid.save(f"tests/TestImageAlign.png")

class TestLandcoverConversion(unittest.TestCase):
    # @unittest.skip("Too slow")
    def test_landcover_conversion(self):
        gray_landcover = np.array(img.open("./data/World_LandCover_512x256.png"))
        rgb_landcover = np.array(img.open("./data/World_LandCover_RGB_512x256.png"))

        converted_landcover = gray_to_land(gray_landcover)
        self.assertTrue(np.array_equal(converted_landcover, rgb_landcover))
        converted_gray_landcover = land_to_gray(rgb_landcover)
        self.assertTrue(np.array_equal(converted_gray_landcover, gray_landcover))


    @unittest.skip("Too slow")
    def test_old_landcover_conversion(self):
        gray_landcover = np.array(img.open("./data/World_LandCover_Mode_512x256.png"))
        rgb_landcover = np.array(img.open("./data/World_LandCover_Mode_RGB_512x256.png"))

        converted_landcover = gray_to_land(gray_landcover, old_classes=True)
        self.assertTrue(np.array_equal(converted_landcover, rgb_landcover))

class TestSpreadFunctions(unittest.TestCase):

    @unittest.skip("Broken")
    def test_spreads(self):
        gray_landcover = np.array(img.open("./data/World_LandCover_512x256.png"))
        spread = continuous_to_spread(gray_landcover, 9)
        continuous = spread_to_continuous(spread)
        self.assertTrue(np.array_equal(continuous, gray_landcover))

class TestColourLists(unittest.TestCase):

    def test_landcover_colours(self):
        gray_landcover = np.array(img.open("./data/World_LandCover_512x256.png"))
        planet_cfg = PlanetConfig()
        landcover_colours = planet_cfg.landcover_colour_list()
        landcover_colours = np.array(landcover_colours)
        real_colours = np.unique(gray_landcover)
        self.assertTrue(np.array_equal(landcover_colours, real_colours))

    def test_sketch_colours(self):
        dem = np.array(img.open("./data/World_DEM_16384x8192.png"))
        planet_cfg = PlanetConfig()
        gray_sketch = dilate_paint(dem, planet_cfg)
        img.fromarray(gray_sketch).save("tests/TestSketchColours_16384x8192.png")
        sketch_colours = planet_cfg.colour_list()
        sketch_colours = np.array(sketch_colours)
        real_colours, counts = np.unique(gray_sketch, return_counts=True)
        self.assertTrue(np.array_equal(sketch_colours, real_colours))

    @unittest.skip("Broken")
    def test_temp_colours(self):
        gray_temp = np.array(img.open("./data/World_Temperature_512x256.png"))
        planet_cfg = PlanetConfig()
        temp_sketch = temperature_paint(gray_temp, planet_cfg)
        temp_colours = planet_cfg.temp_colour_list()
        temp_colours = np.array(temp_colours)
        real_colours = np.unique(temp_sketch)
        self.assertTrue(np.array_equal(temp_colours, real_colours))

class TestEmbedding(unittest.TestCase):

    @unittest.skip("Too slow")
    def test_embedding(self):
        planet_cfg = PlanetConfig()
        # setup(planet_cfg, force=False)
        tile_size = 64
        delta = 32
        down_tile_size = tile_size//delta
        dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
        downland_sketch = dataset.downland_sketch
        downsketch = dataset.downsketch
        downtemp_sketch = dataset.downtemp_sketch
        modal_sketch = dataset.modal_sketch
        num_tests = 10000
        full_h, full_w = downland_sketch.shape
        d = planet_cfg.colours + 1
        l = planet_cfg.landcover_classes + 1
        t = planet_cfg.temp_classes + 1
        for i in tqdm(range(num_tests)):
            y0 = random.randint(0, full_h-down_tile_size)
            x0 = random.randint(0, full_w-down_tile_size)
            metadata = {
                'tile_y': y0*delta,
                'tile_x': x0*delta,
                'k': 0,
                'hflip': 0,
                'vflip': 0,
                'tile_size': tile_size,
                'mask_channel': 4,
            }
            
            embedding = _encode(None, metadata, planet_cfg, downsketch, downland_sketch, downtemp_sketch,
                                            width=full_w, height=full_h)
            
            # Embedding looks like this:
            # Data from 3x3 neighbourhood
            # - Percentage of each elevation colour 9 x d
            # - Percentage of each landcover class 9 x l
            # - Percentage of each temperature class 9 x t
            # - Distance to ocean in all 9 directions 9 x 1 from edges of tile or middle (nearest)
            # This is flattened
            # np.concatenate((
            # sketch_colours.flatten(), 
            # landcover_classes.flatten(), 
            # temp_classes.flatten(),
            # distances))
            get_surrounds = lambda im: _get_tile(im, y0-down_tile_size, x0-down_tile_size, down_tile_size*3)
            real_sketch_surrounds = get_surrounds(downsketch)
            real_land_surrounds = get_surrounds(downland_sketch)
            real_temp_surrounds = get_surrounds(downtemp_sketch)

            sketch_embedding = embedding[:9*d]
            land_embedding = embedding[9*d:9*d+9*l]
            temp_embedding = embedding[9*d+9*l:9*d+9*l+9*t]

            # Reshape embeddings
            sketch_embedding = sketch_embedding.reshape((9, d))
            land_embedding = land_embedding.reshape((9, l))
            temp_embedding = temp_embedding.reshape((9, t))

            # Ensure each column adds to 1.0
            for i in range(9):
                assert np.isclose(np.sum(sketch_embedding[i]), 1.0)
                assert np.isclose(np.sum(land_embedding[i]), 1.0)
                assert np.isclose(np.sum(temp_embedding[i]), 1.0)
            
            # Use percentages to reconstruct colours by first converting to counts
            pixels_per_tile = down_tile_size*down_tile_size
            sketch_counts = (sketch_embedding*pixels_per_tile).astype(np.uint8)
            land_counts = (land_embedding*pixels_per_tile).astype(np.uint8)
            temp_counts = (temp_embedding*pixels_per_tile).astype(np.uint8)

            for i in range(9):
                y = (i // 3) * down_tile_size
                x = (i % 3) * down_tile_size
                get_tile = lambda im: _get_tile(im, y, x, down_tile_size)
                quantize = lambda im, cols: (im/(256/cols)).round().astype(np.uint8)
                sketch_tile = quantize(get_tile(real_sketch_surrounds), d-1)
                land_tile = quantize(get_tile(real_land_surrounds), l-1)
                temp_tile = quantize(get_tile(real_temp_surrounds), t-1)
                emb_sketch_tile_counts = sketch_counts[i]
                emb_land_tile_counts = land_counts[i]
                emb_temp_tile_counts = temp_counts[i]
                sketch_tile_values, sketch_tile_counts = np.unique(sketch_tile, return_counts=True)
                land_tile_values, land_tile_counts = np.unique(land_tile, return_counts=True)
                temp_tile_values, temp_tile_counts = np.unique(temp_tile, return_counts=True)
                full_sketch_tile_counts = np.zeros(d)
                full_land_tile_counts = np.zeros(l)
                full_temp_tile_counts = np.zeros(t)
                full_sketch_tile_counts[sketch_tile_values] = sketch_tile_counts
                full_land_tile_counts[land_tile_values] = land_tile_counts
                full_temp_tile_counts[temp_tile_values] = temp_tile_counts
                sketch_equal = np.array_equal(full_sketch_tile_counts, emb_sketch_tile_counts)
                land_equal = np.array_equal(full_land_tile_counts, emb_land_tile_counts)
                temp_equal = np.array_equal(full_temp_tile_counts, emb_temp_tile_counts)
                assert sketch_equal, f"Sketch: {full_sketch_tile_counts} != {emb_sketch_tile_counts}"
                assert land_equal, f"Land: {full_land_tile_counts} != {emb_land_tile_counts}"
                assert temp_equal, f"Temp: {full_temp_tile_counts} != {emb_temp_tile_counts}"

class TestAutoEncoder(unittest.TestCase):

    def sample(self, vae, name: str):
        planet_cfg = PlanetConfig()
        dataset = RAMDataset(planet_cfg, planet_cfg.output_channels(), planet_cfg.input_channels(), True, 0)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        rows = 10
        grid = []
        t_to_im = lambda t: ToPILImage()(t.squeeze(0))
        for i, item in enumerate(tqdm(dataloader)):
            if i >= rows:
                break
            target = item['target_image']
            with torch.no_grad():
                target_sat = target[:, :3]
                target_dem = target[:, 3:].repeat(1, 3, 1, 1)
                latent_sat = vae.encode(target_sat).latent_dist.sample()
                latent_dem = vae.encode(target_dem).latent_dist.sample()
                output_sat = vae.decoder(latent_sat).clamp(-1, 1)
                output_dem = vae.decoder(latent_dem).clamp(-1, 1)
                row = [output_sat, target_sat, output_dem, target_dem]
                row = list(map(t_to_im, row))
                grid.extend(row)  
        grid = image_grid(grid, rows, 4)
        grid.save(f"tests/TestAutoEncoder-{name}.png")

    @unittest.skip("Too slow")
    def test_stable_diffusion(self):
        repo_id = "stabilityai/stable-diffusion-2-base"
        vae = DiffusionPipeline.from_pretrained(repo_id).vae
        self.sample(vae, "stable")





import warnings
if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        unittest.main()