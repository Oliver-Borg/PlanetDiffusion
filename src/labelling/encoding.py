import torch
import numpy as np

from ..core.utils import normalise_array
from ..core.derivative import elevation_to_SGF
from ..core.shared import ArgsKwargsWrapper
from planetAI.src.data.map_paster import get_tile, setup, get_mask_config, _gen_image, _gen_full_image
from planetAI.src.data.utils import PlanetConfig, image_grid, timing, np_rgb, tensor_to_np
from planetAI.src.data.dataset import RAMDataset, _encode
from torch.utils.data import DataLoader
from os import listdir, mkdir, makedirs
from os.path import join, isdir, exists
from PIL import Image
import json
import re
from typing import Union, Optional
from scipy.ndimage import label
from random import shuffle
import lpips
from tqdm import tqdm
from time import time
from dataclasses import replace
import cv2

# Bins for absolute value of SGFs
SGF_BINS = [
    0.0,
    0.0024977277498692274, 0.00668445834890008,
    0.01155373640358448, 0.01762024126946926,
    0.024745840579271317, 0.032935578376054764,
    0.04226625710725784, 0.053384799510240555,
    0.06668040156364441, 0.08297241479158401,
    0.10330004245042801, 0.1289672553539276,
    0.1617264449596405, 0.2041773498058319,
    0.2624289095401764, 1.0
]

ELEV_BINS = [
    0.0,
    0.07188525050878525, 0.13861295580863953,
    0.19481192529201508, 0.24499885737895966,
    0.29208821058273315, 0.33713284134864807,
    0.3810635507106781, 0.42468908429145813,
    0.46863508224487305, 0.5136644244194031,
    0.5607843399047852, 0.6108644008636475,
    0.6658884286880493, 0.729381263256073,
    0.8109865188598633, 1.0
]

ELEV_RANGE_BINS = [
    # (11.17, 2122.78)
    0, 76.97, 98.8125,
    117.0, 139.22, 165.43,
    212.76, 310.66625, 398.9,
    466.62, 519.06, 574.575,
    635.68, 700.1825, 781.52,
    908.6875, 10000
]

def logify_data(data: np.ndarray) -> np.ndarray:
    '''
    Return a new 1D array filled with the same data as the input array, but
    with the proportion of each element replaced with the log of its count
    This is useful for data which has a very skewed distribution
    '''
    colours, counts = np.unique(data, return_counts=True)
    log_counts = logify_counts(counts)
    total = np.sum(log_counts)
    to_return = np.zeros(data.size, dtype=data.dtype)
    if total == 0:
        return to_return
    offset = data.size//total # TODO: Fix this for channels != 3
    last_i = 0
    for i in range(log_counts.size):
        to_return[last_i:last_i+log_counts[i]*offset] = colours[i]
        last_i += log_counts[i]*offset
    
    return to_return

def logify_counts(counts: np.ndarray) -> np.ndarray:
    '''
    Return the logified counts of the given colours
    '''
    log_counts = np.ceil(np.log10(counts+1)).astype(np.int64)
    return log_counts

def counts_to_data(colours: np.ndarray, counts: np.ndarray) -> np.ndarray:
    '''
    Return a 1D array filled with the given colours, repeated the given number
    '''
    total = np.sum(counts)
    to_return = np.zeros(total, dtype=colours.dtype)
    if total == 0:
        return to_return
    offset = total//colours.size # TODO: Fix this for channels != 3
    last_i = 0
    for i in range(colours.size):
        to_return[last_i:last_i+counts[i]*offset] = colours[i]
        last_i += counts[i]*offset
    return to_return

def remove_zeros(data):
    return data[data != 0]


def compute_bins(data, num_bins):
    # Compute bins so that data follows a uniform distribution
    step_size = 100/num_bins
    percentiles = np.arange(0, 100+step_size, step_size)
    bins = np.percentile(data, percentiles)
    return bins

def compute_count_bins(values, counts, num_bins):
    # Compute bins so that data follows a uniform distribution
    # This is a weighted percentile
    # https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    
    quantiles = np.linspace(0, 1, num_bins+1)
    values = np.array(values)
    weights = np.array(counts)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    weighted_quantiles /= np.sum(weights)
    return np.interp(quantiles, weighted_quantiles, values)


def binify(data, bins):
    # Count number of elements that fall within each bin
    # Return normalised
    height = np.histogram(data, bins)[0]
    total = np.sum(height)
    return height / total if total > 0 else height


def uniform(bins):
    num_items = len(bins) - 1
    return np.ones(num_items) / num_items


class TerrainEncoder:

    @property
    def cross_attention_dim(self):
        raise NotImplementedError

    def baseline(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class TerrainStyle(ArgsKwargsWrapper):

    def __init__(self, terrains, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.terrains = terrains


class GlobalTerrainStyle(TerrainStyle):

    def __init__(self,
                 terrains,
                 ranges,
                 resolutions,
                 *args, **kwargs) -> None:
        super().__init__(terrains, *args, **kwargs)

        self.ranges = ranges
        self.resolutions = resolutions

class PlanetStyle(TerrainStyle):
    
    def __init__(self,
                 sketches,
                 dems,
                 metadata,
                 *args, **kwargs) -> None:
        super().__init__(dems, *args, **kwargs)

        self.sketches = sketches
        self.dems = dems
        self.metadata = []
        for k in metadata:
            for i in range(len(metadata[k])):
                if i >= len(self.metadata):
                    self.metadata.append({})
                self.metadata[i][k] = metadata[k][i]


class SatelliteTerrainStyle(TerrainStyle):
    pass


def generate_features(values, bins):
    features = binify(values, bins)
    offset = uniform(bins)
    return features - offset


class GlobalTerrainEncoder(TerrainEncoder):

    @property
    def cross_attention_dim(self):
        return 48

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)

    def _encode(self, terrain, range, resolution):
        """
        Given a piece of terrain, construct a low dimensional embedding that
        conveys global structure and type.

        Used for:
        - conditioning/encoder_hidden_states for UNet
        - clustering

        (batch, sequence_length, feature_dim)
        - sequence_length: terrain types a user can select (1+)
        - feature_dim: 1D vector where each component encodes some aspect of the terrain
        - [16] unnormalised elevation range distribution/histogram
        - [16] normalised elevation range distribution/histogram
        - [16] transformed slope-angle distribution/histogram
        """

        if terrain.ndim == 3:
            terrain = terrain[0]  # (1, 256, 256) -> (256, 256)

        terrain = range * (terrain + 1)/2  # Scale to (0, max)
        sgf = elevation_to_SGF(terrain, xres=resolution, yres=resolution)

        # Flatten and take absolute value (since terrains can be flipped arbitrarily)
        #  - flattening treats dh_dx and dh_dy the same
        #  - absolute values ignore direction
        sgf = np.abs(sgf.flatten())

        return self._generate_features(terrain, sgf)

    def _generate_features(self, terrain, sgf):
        features_1 = generate_features(terrain, ELEV_RANGE_BINS)
        features_2 = generate_features(normalise_array(terrain), ELEV_BINS)
        features_3 = generate_features(sgf, SGF_BINS)

        # Concatenate to form a single feature vector
        return np.concatenate((features_1, features_2, features_3)).astype(np.float32)

    def _prepare_data(self, terrain_style):
        to_return = []

        for i in (terrain_style.terrains, terrain_style.ranges, terrain_style.resolutions):
            if isinstance(i, torch.Tensor):
                i = i.cpu().numpy()
            to_return.append(i)

        return to_return

    def __call__(self, terrain_style: GlobalTerrainStyle):
        # NOTE: all variables are batched
        terrains, ranges, resolutions = self._prepare_data(terrain_style)
        batched_encodings = []
        for terrain, range, resolution in zip(terrains, ranges, resolutions):
            enc = self._encode(terrain, range, resolution)
            batched_encodings.append(torch.from_numpy(enc).unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings


SATELLITE_HIST_R = [
    0.0, 0.12156862745098039, 0.15294117647058825,
    0.17647058823529413, 0.19607843137254902, 0.21568627450980393,
    0.23529411764705882, 0.25882352941176473, 0.2823529411764706,
    0.3137254901960784, 0.35294117647058826, 0.403921568627451,
    0.4666666666666667, 0.5294117647058824, 0.592156862745098,
    0.6588235294117647, 1.0
]

SATELLITE_HIST_G = [
    0.0, 0.19215686274509805, 0.22745098039215686,
    0.24705882352941178, 0.26666666666666666, 0.2823529411764706,
    0.2980392156862745, 0.3137254901960784, 0.3333333333333333,
    0.35294117647058826, 0.3803921568627451, 0.4117647058823529,
    0.4549019607843137, 0.5058823529411764, 0.5529411764705883,
    0.611764705882353, 1.0
]

SATELLITE_HIST_B = [
    0.0, 0.10588235294117647, 0.12549019607843137,
    0.1411764705882353, 0.1568627450980392, 0.17254901960784313,
    0.19215686274509805, 0.21176470588235294, 0.23137254901960785,
    0.2549019607843137, 0.2784313725490196, 0.3137254901960784,
    0.35294117647058826, 0.396078431372549, 0.44313725490196076,
    0.49411764705882355, 1.0
]


class PlanetEncoder(TerrainEncoder):
    
    def __init__(self, planet_cfg, sketch_factor=1, land_factor=1, distance_factor=1):
        self.colours = planet_cfg.sketch_colours
        self.classes = planet_cfg.landcover_classes
        self.temps = planet_cfg.temp_classes
        self.planet_cfg = planet_cfg
        self.sketch_factor = sketch_factor
        self.land_factor = land_factor
        self.distance_factor = distance_factor
    
    @property
    def cross_attention_dim(self):
        # Add an extra element for ocean for each type
        return 9 * (self.colours + 1 + self.classes + 1 + self.temps + 1) + 9

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)  

    def _prepare_data(self, terrain_style):
        to_return = []

        for i in (terrain_style.sketches, terrain_style.metadata):
            if isinstance(i, torch.Tensor):
                i = i.cpu().numpy()
            to_return.append(i)

        return to_return

    def __call__(self, terrain_style: PlanetStyle):
        # NOTE: all variables are batched
        sketches, metadatas = self._prepare_data(terrain_style)
        batched_encodings = []
        for sketch, metadata in zip(sketches, metadatas):
            enc = _encode(None, metadata, self.planet_cfg, None, None, None)
            if type(enc) == np.ndarray:
                enc = torch.from_numpy(enc)
            batched_encodings.append(enc.unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings




class OldPlanetEncoder(TerrainEncoder):

    def __init__(self, planet_cfg, dem_encoding = True, log_encoding = True, 
                 use_latitude = False, force = False, normalise_features = False, num_workers = 0):
        #TODO: Set latitude to True and test training
        '''
        Optimal settings for different sizes
        - 0: dem_encoding=True, log_encoding=True
        - 1: dem_encoding=True, log_encoding=True
        - 2: dem_encoding=True, log_encoding=True
        - 3: dem_encoding=N/A, log_encoding=N/A
        - 4: dem_encoding=N/A, log_encoding=N/A
        - 5: dem_encoding=N/A, log_encoding=N/A
        '''
        super().__init__()
        self.planet_cfg = planet_cfg
        self.bins = get_planet_bins(planet_cfg, dem_encoding, log_encoding, force, num_workers=num_workers)
        self.dem_encoding = dem_encoding
        self.log_encoding = log_encoding
        self.use_latitude = use_latitude
        self.normalise_features = normalise_features

    @property
    def cross_attention_dim(self):
        return 48 + int(self.use_latitude) * 2**self.planet_cfg.size

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)

    def _encode(self, sketch, dem, tile_y):
        """
        Given a piece of terrain, construct a low dimensional embedding that
        conveys global structure and type.

        Used for:
        - conditioning/encoder_hidden_states for UNet
        - clustering

        (batch, sequence_length, feature_dim)
        - sequence_length: terrain types a user can select (1+)
        - feature_dim: 1D vector where each component encodes some aspect of the terrain
        dem_encoding True:
        - [16] unnormalised elevation range distribution/histogram
        - [16] normalised elevation range distribution/histogram
        - [16] transformed slope-angle distribution/histogram

        dem_encoding False:
        - [16] unnormalised sketch elevation range distribution/histogram
        - [16] unnormalised DEM elevation range distribution/histogram
        - [16] unnormalised river distribution/histogram

        use_latitude True:
        - [2] latitude
        """

        sketch = 255 * (sketch + 1)/2  # Scale to (0, 255)
        dem = 255 * (dem + 1)/2
        # Transpose from torch (C, H, W) to numpy (H, W, C)
        if np.argmin(sketch.shape) == 0:
            sketch = np.transpose(sketch, (1, 2, 0))
        if np.argmin(dem.shape) == 0:
            dem = np.transpose(dem, (1, 2, 0))


        return self._generate_features(sketch, dem, self.bins, tile_y)

    def _generate_features(self, sketch, dem, bins, tile_y):
        features = list(get_features(sketch, dem, self.dem_encoding))
        if self.log_encoding:
            features[:2] = [generate_features(logify_data(feature), bins[i]) for i, feature in enumerate(features[:2])]
            # We don't logify the gradient/river values since they aren't badly skewed
            features[2] = generate_features(features[2], bins[2]) 
        else:
            features = [generate_features(feature, bins[i]) for i, feature in enumerate(features)]
        if self.use_latitude:
            one_hot = np.zeros(2**self.planet_cfg.size)
            one_hot[tile_y] = 1
            features.append(one_hot)
        if self.normalise_features:
            for feature in features:
                feature = np.array(feature)
                feature /= np.max(feature)
        return np.concatenate(features).astype(np.float32)

    def _prepare_data(self, terrain_style):
        to_return = []

        for i in (terrain_style.sketches, terrain_style.dems, terrain_style.metadata):
            if isinstance(i, torch.Tensor):
                i = i.cpu().numpy()
            to_return.append(i)

        return to_return

    def __call__(self, terrain_style: PlanetStyle):
        # NOTE: all variables are batched
        sketches, dems, metadatas = self._prepare_data(terrain_style)
        batched_encodings = []
        for sketch, dem, tile_y in zip(sketches, dems, metadatas['tile_y']):
            enc = self._encode(sketch, dem, tile_y)
            batched_encodings.append(torch.from_numpy(enc).unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings


class SatelliteTerrainEncoder(TerrainEncoder):

    @property
    def cross_attention_dim(self):
        return 48

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)

    def _encode(self, terrain):
        if isinstance(terrain, torch.Tensor):
            terrain = terrain.cpu().numpy()

        # Convert (-1, 1) to (0, 1)
        terrain = (terrain + 1)/2
        
        features_1 = generate_features(terrain[0].flatten(), SATELLITE_HIST_R)
        features_2 = generate_features(terrain[1].flatten(), SATELLITE_HIST_G)
        features_3 = generate_features(terrain[2].flatten(), SATELLITE_HIST_B)

        # Concatenate to form a single feature vector
        return np.concatenate((features_1, features_2, features_3)).astype(np.float32)

    def __call__(self, terrain_style: SatelliteTerrainStyle):
        # NOTE: all variables are batched
        batched_encodings = []
        for terrain in terrain_style.terrains:
            enc = self._encode(terrain)
            batched_encodings.append(torch.from_numpy(enc).unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings


def get_histogram(bin_dir, channel):
    bin_file = join(bin_dir, f'{channel}.json')
    if not exists(bin_file):
        return None
    with open(bin_file, 'r') as f:
        dist = json.load(f)['bins']
        return dist

def make_default_bins(data_dir, channels, force=False):
    bin_dir = join(data_dir, f'_bins')
    if not exists(bin_dir):
        makedirs(bin_dir)
    bin_files = 0
    for i in range(channels):
        bin_file = join(bin_dir, f'{i}.json')
        if exists(bin_file):
            bin_files += 1
    if bin_files == channels and not force:
        return
    tiles = listdir(data_dir)
    if "_empty" in tiles:
        tiles.remove("_empty")
    data = [[] for _ in range(channels)]
    for tile in tiles:
        tile_dir = join(data_dir, tile)
        if not isdir(tile_dir) or not re.match(r"^\d+_\d+_\d+_\d+_\d+_\d+$", tile):
            print(f"Skipping {tile}")
            continue
        tile_dir = join(tile_dir, "dem.jpg")
        img = Image.open(tile_dir)
        img = np.array(img)
        for i in range(channels):
            data[i] += list(img[:,:,i].flatten() / 255.0)
    for i in range(channels):
        stats = {}
        stats['bins'] = list(compute_bins(data[i], 16))
        stats['min'] = min(data[i])
        stats['max'] = max(data[i])
        stats['mean'] = np.mean(data[i])
        stats['std'] = np.std(data[i])
        
        with open(join(bin_dir, f'{i}.json'), "w") as f:
            json.dump(stats, f, indent=4)

def get_features(sketch, dem, dem_encoding):

    if len(dem.shape) == 3:
        dem_tile = dem[:,:,0]
    else:
        dem_tile = dem
    if dem_encoding:
        features_1 = dem_tile.flatten()
        features_2 = normalise_array(dem_tile).flatten()
        sgf = elevation_to_SGF(dem_tile, 1, 1, dem_tile > 0)
        sgf = sgf.flatten()
        # Remove nan values
        sgf = sgf[~np.isnan(sgf)]
        features_3 = np.abs(sgf.flatten())
    else:
        features_1 = sketch[:, :, 0].flatten()
        features_2 = dem_tile.flatten()
        labeled, num_components = label(sketch[:, :, 1])
        features_3 = np.bincount(labeled.flatten())[1:]
    
    return features_1, features_2, features_3

def get_planet_bins(planet_cfg: PlanetConfig, dem_encoding: bool=True, log_encoding: bool=True, force: bool=False, num_workers: int=0) -> list[list[float]]:
    '''
    Generate bins to use for the planet encoder
    Args:
        planet_cfg: The planet config
        dem_encoding: Whether to use only the dem for encoding or to use the dem, sketch and rivers
        log_encoding: Whether to use logify_counts to reduce the impact of zeroes
        force: Whether to force regeneration of the bins
        num_workers: The number of workers to use
    Returns:
        A list of lists of bins
    '''
    iters = planet_cfg.iters
    size = planet_cfg.size

    bins = {}
    bin_dir = join(planet_cfg.transformation_dir(), 'bins')
    makedirs(bin_dir, exist_ok=True)
    bin_dir = join(bin_dir, f'{str(planet_cfg)}-DE{str(dem_encoding)[0]}-LE{str(log_encoding)[0]}.json')
    if exists(bin_dir) and not force:
        with open(bin_dir, 'r') as f:
            bins = json.load(f)
    if str(size) in bins:
        return bins[str(size)]
    # TODO: Change this if planet_cfg.gen_lists is False
    # TODO Test this
    new_cfg = replace(planet_cfg, iters=min(10000, iters))
    # to_generate = efficient_tiles(to_generate)
    setup(new_cfg)
    dataset = RAMDataset(new_cfg)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    h, w = 256*2**size, 512*2**size
    
    # Data for each feature channel we want to encode
    # 0: sketch, 1: dem, 2: rivers
    features_dicts = [{}, {}, {}]

    i = 0
    # with tqdm(total=new_cfg.iters, desc="Generating encoder bins") as pbar:
    #     for item in dataloader:
    for item in tqdm(dataloader, total=new_cfg.iters, desc="Generating encoder bins"):

        sketch_tile = tensor_to_np_img(item['cond_image'][0])
        dem_tile = tensor_to_np_img(item['target_image'][0])
        features = get_features(sketch_tile, dem_tile, dem_encoding)
        for f, feature in enumerate(features):
            f_vals, f_counts = np.unique(feature, return_counts=True)
            for val, count in zip(f_vals, f_counts):
                if val not in features_dicts[f]:
                    features_dicts[f][val] = 0
                features_dicts[f][val] += count
                i+=1
    # We use the log of the counts to reduce the impact of the many zeros
    # Removing the zeros meant losing information about the distribution
    # We don't logify the gradient/river values since they aren't badly skewed
    if log_encoding:
        for f in range(2):
            logged_counts = logify_counts(np.array(list(features_dicts[f].values())))
            features_dicts[f] = dict(zip(features_dicts[f].keys(), logged_counts))
            
            
    to_return = [
        list(compute_count_bins(np.array(list(features_dicts[f].keys())), 
                           np.array(list(features_dicts[f].values())), 
                           16)) for f in range(3)
    ]

    # Options:
    # - Component sizes
    # - Number of rivers (only one value per sketch)
    # - River width at each point


   
    bins[str(size)] = to_return

    with open(bin_dir, 'w') as f:
        json.dump(bins, f)

    return to_return

def tensor_to_np_img(x) -> np.ndarray:
    '''
    Return the given tensor as a numpy image
    '''
    if len(x.shape) == 4:
        x = x[0]
    return ((x.cpu().detach().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)

def kmeans(data: np.ndarray, k: int) -> tuple:
    '''
    Run kmeans on the given data
    Args:
        data: The data to cluster
        k: The number of clusters to use
    Returns:
        A list of lists of tiles, where each list is a cluster of indexes
    '''

    assert len(data.shape) == 2
    assert data.shape[0] >= k
    assert data.max() != np.nan
    assert data.min() != np.nan
    changed = True

    centroids = data[np.random.choice(data.shape[0], k, replace=False),:]

    while changed:
        changed = False
        clusters = [[] for _ in range(k)]
        for i in range(data.shape[0]):
            dists = [np.sum((data[i,:] - c)**2) for c in centroids]
            closest = np.argmin(dists)
            clusters[closest].append(i)
        for i in range(k):
            new_centroid = np.mean(data[clusters[i],:], axis=0) if len(clusters[i]) > 0 else centroids[i]
            if not np.array_equal(new_centroid, centroids[i]):
                changed = True
                centroids[i] = new_centroid

    dists = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            dists[i, j] = np.sum((centroids[i] - centroids[j])**2)

    return clusters, dists, centroids

def get_cluster_metrics(k: int, num_tiles: int, clusters: list, metric_fns: tuple) -> list:
    '''
    Get the metrics for the given clusters
    Args:
        k: The number of clusters
        num_tiles: The number of tiles
        clusters: The clusters to get metrics for
        metrics: The metrics to use
    Returns:
        A list of metrics for each cluster
    '''
    metrics = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for cluster in range(k):
        scores = [[] for i in range(len(metric_fns))]
        times = [[] for i in range(len(metric_fns))]
        for col in range(num_tiles):
            _, _, dem0 = clusters[cluster * num_tiles + col]
            if dem0.sum() == 0:
                continue
            ten0 = torch.tensor((dem0.astype(np.float32)/255*2-1).transpose(2, 0, 1)).to(device)
            for n in range(col+1, num_tiles):
                _, _, dem1 = clusters[cluster * num_tiles + n]
                if dem1.sum() == 0:
                    continue
                ten1 = torch.tensor((dem1.astype(np.float32)/255*2-1).transpose(2, 0, 1)).to(device)
                for i, metric_fn in enumerate(metric_fns):
                    t0 = time()
                    scores[i].append(metric_fn(ten0, ten1).item())
                    times[i].append(time() - t0)
        if len(scores[0]) > 0:
            metrics.append([np.mean(scores, axis=1), np.mean(times, axis=1)])
    return np.mean(metrics, axis=0).flatten()

def cluster(planet_cfg: PlanetConfig, planet_encoder: PlanetEncoder, k: int, num_tiles: int = 1, num_workers: int=0) -> tuple:
    '''
    Cluster the planet data based on the given encoder
    Args:
        planet_cfg: The planet config
        planet_encoder: The planet encoder
        k: The number of clusters to use
        num_tiles: The number of tiles to return per cluster
        num_workers: The number of workers to use
    Returns:
        A list of k tiles
    '''
    dataset = RAMDataset(planet_cfg)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    data = np.zeros((len(dataset), planet_encoder.cross_attention_dim), dtype=np.float32)
    i = 0
    dataset_images = []
    for item in tqdm(dataloader, total=len(dataset), desc="Clustering"):
        sketch_tile = item['cond_image']
        dem_tile = item['target_image']
        style = PlanetStyle(sketch_tile, dem_tile, item['metadata'])
        sketch_tile = tensor_to_np_img(item['cond_image'])[:, :, 5:6]
        land_sketch_tile = tensor_to_np_img(item['cond_image'])[:, :, 6:7]
        dem_tile = tensor_to_np_img(item['target_image'])[:, :, 0:1]
        features = planet_encoder(style)
        data[i] = features.cpu().numpy().copy()
        dataset_images.append((sketch_tile, land_sketch_tile, dem_tile))
        i+=1 

    clusters, dists, centroids = kmeans(data, k)

    to_return = []
    
    for i in range(k):
        shuffle(clusters[i])
        for n in range(num_tiles):
            if len(clusters[i]) <= n:
                to_return.append((np.zeros((256, 256, 3), dtype=np.uint8), 
                                  np.zeros((256, 256, 3), dtype=np.uint8),
                                  np.zeros((256, 256, 3), dtype=np.uint8)))
                continue
            want = clusters[i][n]
            to_return.append(dataset_images[want])

    return to_return, dists, centroids
    
def main():
    
    it = 1000
    num_tiles = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    mse = torch.nn.MSELoss().to(device)
    size = 5
    do = 5
    for _ in range(1):
        default_cfg = PlanetConfig(iters=it, size=size, data_dir='./planetAI/data', 
                                   image_mode='planet', downscale_offset=do,
                                   dilate_iters=0, erode_iters=0)
        setup(default_cfg)
        records = []
        with tqdm(total=2) as pbar:
            for sf in [0, 1, 2]:
                for lf in [0, 1, 2]:
                    for df in [0, 1, 2]:
                        pbar.set_description("Preparing encoder")
                        encoder = PlanetEncoder(default_cfg, sketch_factor=sf, land_factor=lf, distance_factor=df)
                        pbar.set_description("Setting up planet config")
                        planet_cfg = replace(default_cfg)
                        setup(planet_cfg)
                        for k in [8, 16]:
                            pbar.set_description(f'Clustering {k} clusters')
                            imgs, dists, centroids = cluster(planet_cfg, encoder, k, num_tiles, num_workers=0)
                            grid = []
                            texts = []
                            pbar.set_description('Getting metrics')
                            lpips_score, mse_score, lpips_time, mse_time = get_cluster_metrics(k, num_tiles, imgs, (lpips_model, mse))
                            records.append([size, k, sf, lf, df, lpips_score, mse_score, lpips_time, mse_time])
                            for i, (sketch, land_sketch, dem) in enumerate(imgs):
                                # texts.append(f'Cluster {i//num_tiles}')
                                
                                sketch = Image.fromarray(sketch[:, :, 0])
                                land_sketch = Image.fromarray(land_sketch[:, :, 0])
                                dem = Image.fromarray(np_rgb(dem[:, :, 0], 'terrain', planet_cfg.get_max_pixel(dem)))
                                # grid.append(sketch)
                                grid.append(sketch)
                                grid.append(land_sketch)
                                grid.append(dem)
                                texts.append(f'LPIPs:{lpips_score:0.4f}|MSE:{mse_score:0.4f}')
                            grid = image_grid(grid, k, 3*num_tiles, row_labels=[f'C{c}' for c in centroids])
                            makedirs(f'./planetAI/data/test/cluster_SZ{size}', exist_ok=True)
                            grid.save(f'./planetAI/data/test/cluster_SZ{size}/K{k}_SF{sf}_LF{lf}_DF{df}.png')
                            with open(f'./planetAI/data/test/cluster_SZ{size}/K{k}_SF{sf}_LF{lf}_DF{df}.txt', 'w') as f:
                                f.writelines(texts)
                                f.write(f"Mean inter-cluster distance: {str(dists.mean())}\n")
                                print(f"Mean inter-cluster distance: {str(dists.mean())}")
                                f.write("All inter-cluster distances:\n")
                                f.write(str(dists))
                            pbar.update(1)
        print("Lowest LPIPs:")
        records.sort(key=lambda x: x[5])
        for r in records[:5]:
            size, k, sf, lf, df, lpips_score, mse_score, lpips_time, mse_time = r
            print(f'Size: {size}, K: {k}, SF: {sf}, LF: {lf}, DF: {df}, Avg LPIPS: {lpips_score:0.4f}({1000*lpips_time:0.2f}ms), Avg MSE: {mse_score:0.4f}({1000*mse_time:0.2f}ms)') 
        
        print("Lowest MSE:")
        records.sort(key=lambda x: x[6])
        for r in records[:5]:
            size, k, sf, lf, df, lpips_score, mse_score, lpips_time, mse_time = r
            print(f'Size: {size}, K: {k}, SF: {sf}, LF: {lf}, DF: {df}, Avg LPIPS: {lpips_score:0.4f}({1000*lpips_time:0.2f}ms), Avg MSE: {mse_score:0.4f}({1000*mse_time:0.2f}ms)')
if __name__ == "__main__":
    main()