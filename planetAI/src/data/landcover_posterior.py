from .dataset import RAMDataset
from .utils import PlanetConfig, tensor_to_np
from torch.utils.data import DataLoader
import numpy as np
from scipy.ndimage import generic_filter
from tqdm import tqdm
from PIL import Image
from numpy.random import choice
import os

def count_kernel(a: np.ndarray) -> np.ndarray:
    """
    Given a list of 8 pixels, return the count of each class
    """
    counts = [0 for i in range(9)]
    for i in a:
        counts[i] += 1
    return counts

def name_kernel(a: np.ndarray) -> int:
    """
    Given a list of 8 pixels, return the name of the combination of pixels
    """
    name = 0
    # TODO Maybe don't sort
    a.sort()
    for i in a:
        name = name * 9 + int(i)
    return name

def p_prediction_f(p: np.ndarray) -> int:
    """
    Given a list of probabilities, return the class based on the probabilities
    """
    if np.sum(p) == 0:
        return 0
    return choice([i for i in range(9)], p=p)

def m_prediction_f(p: np.ndarray) -> int:
    """
    Given a list of probabilities, return the class based on the probabilities
    """
    # TODO: Possibly use probabilites if the max is less than a threshold
    return np.argmax(p)

def save_counts(counts: dict, filename: str):
    """
    Save the counts to a file
    """
    # We could just save the entire array but it's a bit big
    # Instead we should save tuples of (index, count) if count > 0
    with open(filename, 'w') as f:
        for k in counts:
            f.write(f"{int(k)} {int(counts[k])}\n")

def load_counts(filename: str) -> dict:
    """
    Load the counts from a file
    """
    # TODO Add empty class so we can predict when there is less than 8 known pixels
    # counts = np.zeros((9**10), dtype=np.uint16)
    counts = {}
    if not os.path.exists(filename):
        return counts
    n = 0
    with open(filename, 'r') as f:
        for line in f:
            i, c = line.split()
            counts[int(i)] = int(c)
            n += 1
    print(f"Loaded {n} counts")
    return counts

if __name__ == "__main__":
    # We want to find the probability of a pixel belonging to a class given it's 
    # surrounding 8 pixels and the class of the corresponding pixel in the sketch
    # There are 9 classes so we can use [1-9] to encode the classes.
    # For simplicity we can just index these by a 9 digit string.
    # counts = np.array([[0 for k in range(9)] for i in range(10**9)])
    counts = load_counts('counts.txt')
    

    planet_cfg = PlanetConfig(image_mode='land', size=5, downscale_offset=5)
    train_dataset = RAMDataset(planet_cfg, mode='train')
    val_dataset = RAMDataset(planet_cfg, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    epochs = 10
    sketch_channel = planet_cfg.input_index('downland_sketch')
    output_channel = planet_cfg.output_index('land')
    colour_step = (255 // planet_cfg.landcover_classes)

    

    # Alternative is to sort the 8 pixels so they are position independent

    for epoch in range(epochs):
        for data in tqdm(train_dataloader):
            target_img = data['target_image']
            cond_image = data['cond_image']
            land_sketch = cond_image[0, sketch_channel]
            land_target = target_img[0, output_channel]
            land_sketch = tensor_to_np(land_sketch)
            land_target = tensor_to_np(land_target)
            land_sketch //= colour_step
            land_target //= colour_step
            w, h = land_sketch.shape
            # Take a 3x3 kernel over the land_target and return a list of pixels in the kernel for each pixel in the land_target
            # This should be vectorised for numpy
            footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            names = generic_filter(land_target.astype(np.float64), name_kernel, footprint=footprint).astype(np.uint32)
            names = names * 9 + land_sketch
            names = names * 9 + land_target
            indices, num = np.unique(names, return_counts=True) 
            for i, n in zip(indices, num):
                counts[i] = counts.get(i, 0) + int(n)
            # TODO: Maybe do this:
            # np.vectorize(lambda x: counts.update({x: counts.get(x, 0) + 1}))(names)

        p_correct_pixels = 0
        m_correct_pixels = 0
        total_pixels = 0
        for data in tqdm(val_dataloader):
            target_img = data['target_image']
            cond_image = data['cond_image']
            land_sketch = cond_image[0, sketch_channel]
            land_target = target_img[0, output_channel]
            land_sketch = tensor_to_np(land_sketch)
            land_target = tensor_to_np(land_target)
            land_sketch //= colour_step
            land_target //= colour_step
            w, h = land_sketch.shape
            # Take a 3x3 kernel over the land_target and return a list of pixels in the kernel for each pixel in the land_target
            # This should be vectorised for numpy
            footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            names = generic_filter(land_target.astype(np.float64), name_kernel, footprint=footprint).astype(np.uint32)
            names = names * 9 + land_sketch
            names = names * 9
            # We want to make a prediction for each pixel in the land_target
            # We can use the counts to make a prediction
            count_stack = np.array([names + i for i in range(9)])
            for i in range(9):
                count_stack[i] = np.vectorize(lambda x: counts.get(x, 0))(count_stack[i])
            prob_stack = count_stack / np.sum(count_stack, axis=0)
            # Replace nan with 0
            prob_stack = np.nan_to_num(prob_stack)
            p_prediction = np.apply_along_axis(p_prediction_f, 0, prob_stack)
            m_prediction = np.apply_along_axis(m_prediction_f, 0, prob_stack)
            p_correct_pixels += np.sum(p_prediction == land_target)
            m_correct_pixels += np.sum(m_prediction == land_target)
            total_pixels += w * h
        print(f"Epoch {epoch} probability accuracy: {p_correct_pixels / total_pixels if total_pixels > 0 else 0.0}")
        print(f"Epoch {epoch} argmax accuracy: {m_correct_pixels / total_pixels if total_pixels > 0 else 0.0}")
        save_counts(counts, 'counts.txt')

            








