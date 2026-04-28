import numpy as np
import numpy.typing as npt
from typing import Optional
import cv2
import os
import PIL.Image as img
import maxflow

from math import sqrt
from random import randint
import networkx as nx
import matplotlib.pyplot as plt

from time import time
try:
    from utils import show_imgs, PlanetConfig, image_grid

    from map_paster import setup, gen_tile_pair, get_tile, set_tile
except:
    from .utils import show_imgs, PlanetConfig, image_grid

    from .map_paster import setup, gen_tile_pair, get_tile, set_tile
# From https://github.com/niranjantdesai/image-blending-graphcuts
class GraphCuts:
    """
    Main class for image synthesis with graph cuts
    """
    def __init__(self, src, sink, mask, save_graph=False):
        """
        Initializes the graph and computes the min-cut.
        :param src: image to be blended (foreground)
        :param sink: background image
        :param mask: manual mask with constrained pixels
        :param save_graph: if true, graph is saved
        """
        assert (src.shape == sink.shape), \
            f"Source and sink dimensions must be the same: {str(src.shape)} != {str(sink.shape)}"

        # Create the graph
        graph = maxflow.Graph[float]()
        # Add the nodes. node_ids has the identifiers of the nodes in the grid.
        node_ids = graph.add_grid_nodes((src.shape[0], src.shape[1]))

        self.compute_edge_weights(src, sink)

        # Add non-terminal edges
        # TODO: use alternate API which is more efficient
        patch_height = src.shape[0]
        patch_width = src.shape[1]
        for row_idx in range(patch_height):
            for col_idx in range(patch_width):
                # right neighbor
                if col_idx + 1 < patch_width:
                    weight = self.edge_weights[row_idx, col_idx, 0]
                    graph.add_edge(node_ids[row_idx][col_idx],
                                   node_ids[row_idx][col_idx + 1],
                                   weight,
                                   weight)

                # bottom neighbor
                if row_idx + 1 < patch_height:
                    weight = self.edge_weights[row_idx, col_idx, 1]
                    graph.add_edge(node_ids[row_idx][col_idx],
                                   node_ids[row_idx + 1][col_idx],
                                   weight,
                                   weight)

                # Add terminal edge capacities for the pixels constrained to
                # belong to the source/sink.
                if np.array_equal(mask[row_idx, col_idx, :], [0, 255, 255]):
                    graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
                elif np.array_equal(mask[row_idx, col_idx, :], [255, 128, 0]):
                    graph.add_tedge(node_ids[row_idx][col_idx], np.inf, 0)

        # Plot graph
        if save_graph:
            nxg = graph.get_nx_graph()
            self.plot_graph_2d(nxg, patch_height, patch_width)

        # Compute max flow / min cut.
        flow = graph.maxflow()
        self.sgm = graph.get_grid_segments(node_ids)

    def compute_edge_weights(self, src, sink):
        """
        Computes edge weights based on matching quality cost.
        :param src: image to be blended (foreground)
        :param sink: background image
        """
        self.edge_weights = np.zeros((src.shape[0], src.shape[1], 2))

        # Create shifted versions of the matrices for vectorized operations.
        src_left_shifted = np.roll(src, -1, axis=1)
        sink_left_shifted = np.roll(sink, -1, axis=1)
        src_up_shifted = np.roll(src, -1, axis=0)
        sink_up_shifted = np.roll(sink, -1, axis=0)

        # Assign edge weights.
        # For numerical stability, avoid divide by 0.
        eps = 1e-10

        # Right neighbor.
        weight = np.sum(np.square(src - sink, dtype=np.float32) +
                        np.square(src_left_shifted - sink_left_shifted, 
                        dtype=np.float32),
                        axis=2)
        norm_factor = np.sum(np.square(src - src_left_shifted, dtype=np.float32) +
                             np.square(sink - sink_left_shifted, 
                             dtype=np.float32),
                             axis=2)
        self.edge_weights[:, :, 0] = weight / (norm_factor + eps)

        # Bottom neighbor.
        weight = np.sum(np.square(src - sink, dtype=np.float32) +
                        np.square(src_up_shifted - sink_up_shifted,
                        dtype=np.float32),
                        axis=2)
        norm_factor = np.sum(np.square(src - src_up_shifted, dtype=np.float32) +
                             np.square(sink - sink_up_shifted, 
                             dtype=np.float32),
                             axis=2)
        self.edge_weights[:, :, 1] = weight / (norm_factor + eps)

    def plot_graph_2d(self, graph, nodes_shape, plot_weights=False, 
                      plot_terminals=True, font_size=7):
        """
        Plot the graph to be used in graph cuts
        :param graph: PyMaxflow graph
        :param nodes_shape: patch shape
        :param plot_weights: if true, edge weights are shown
        :param plot_terminals: if true, the terminal nodes are shown
        :param font_size: text font size
        """
        X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}
        positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
        positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

        nxgraph = graph.get_nx_graph()
        print("nxgraph created")
        if not plot_terminals:
            nxgraph.remove_nodes_from(['s', 't'])

        plt.clf()
        nx.draw(nxgraph, pos=positions)

        if plot_weights:
            edge_labels = {}
            for u, v, d in nxgraph.edges(data=True):
                edge_labels[(u, v)] = d['weight']
            nx.draw_networkx_edge_labels(nxgraph,
                                         pos=positions,
                                         edge_labels=edge_labels,
                                         label_pos=0.3,
                                         font_size=font_size)

        plt.axis('equal')
        plt.show()

    def blend(self, src, target):
        """
        Blends the target image with the source image based on the graph cut.
        :param src: Source image
        :param target: Target image
        """
        target[self.sgm] = src[self.sgm]
        return target



def graph_cut_merge(tile0: npt.NDArray[np.float32], tile1: npt.NDArray[np.float32], updown: bool, 
                    sides: int=0b1111) -> npt.NDArray[np.float32]:
    '''
    This is a greedy algorithm that tries to find the best path to merge two overlapping tiles.
    '''
    to_return = np.zeros_like(tile0)
    tile1 = (np.dstack([tile1, tile1, tile1])*255).astype(np.uint8)
    tile0 = (np.dstack([tile0, tile0, tile0])*255).astype(np.uint8)
    mask = np.zeros_like(tile0)
    if updown:
        mask[mask.shape[0]//4*3, :, :] = [255, 128, 0]
        mask[mask.shape[0]//4, :, :] = [0, 255, 255]
    else:
        mask[:, mask.shape[1]//4*3, :] = [255, 128, 0]
        mask[:, mask.shape[1]//4, :] = [0, 255, 255]
        
    graph_cuts = GraphCuts(tile1, tile0, mask)
    cut = graph_cuts.sgm

    

    to_return[cut] = 1.
    if sides & 0b0001 or sides & 0b0100:
        to_return = 1 - to_return


    return to_return

    if updown:
        tile0 = np.transpose(tile0, (1, 0))
        tile1 = np.transpose(tile1, (1, 0))
    
    
    diffs = np.abs(np.roll(tile0, -1, axis=0) - tile1)
    print(tile0, tile1, diffs, sep='\n\n')
    last_pos = np.argmin(diffs[:, 0])
    to_return = np.zeros_like(tile0)
    to_return[:last_pos, 0] = 1.
    for i in range(1, diffs.shape[1]):
        l = max(0, last_pos - 1)
        r = min(diffs.shape[0], last_pos + 2)
        next_pos = np.argmin(diffs[l:r, i])
        to_return[:last_pos, i] = 1.
        last_pos += next_pos - 1

    if updown:
        return np.transpose(to_return, (1, 0))
    else:
        return to_return


def poisson_blending_merge(tile0: npt.NDArray[np.float32], tile1: npt.NDArray[np.float32], updown: bool) -> npt.NDArray[np.float32]:
    '''
    Return the blended tile
    Args:
        tile0: The first full tile
        tile1: The second full tile
        updown: Whether the tiles are overlapping vertically or horizontally
    '''
    tile0 = np.dstack([tile0]*3)
    tile1 = np.dstack([tile1]*3)
    dst_h = tile1.shape[0]
    dst_w = tile1.shape[1]
    if updown:
        dst_h *= 2
    else:
        dst_w *= 2
    overlap_width = tile0.shape[0] - tile1.shape[0] if updown else tile0.shape[1] - tile1.shape[1]
    dst = np.zeros((dst_h, dst_w, 3), np.float32)
    dst[:tile0.shape[0], :tile0.shape[1], :] = tile0
    mask = np.ones_like(tile1, np.uint8)*255
    if updown:
        dst[tile0.shape[0]:tile0.shape[0]+tile1.shape[0]-overlap_width, :tile1.shape[1], :] = tile1[overlap_width:, :, :]
        # mask[:overlap_width, :, :] = 0
        center = (dst.shape[1]//2, dst.shape[0]//2)
        dst = cv2.seamlessClone(tile1, dst, mask, center, cv2.NORMAL_CLONE)
    else:
        dst[:tile1.shape[0], tile0.shape[1]+overlap_width:tile0.shape[1]+tile1.shape[1], :] = tile1[:, overlap_width:, :]
        # mask[:, :overlap_width, :] = 0
        center = (tile0.shape[1]+tile0.shape[1]//2, tile0.shape[0]//2)
        dst = cv2.seamlessClone(tile1, dst, mask, center, cv2.MIXED_CLONE)
    dst = dst[:, :]
    return dst

def alpha_blending_merge(tile0: npt.NDArray[np.float32], tile1: npt.NDArray[np.float32], updown: bool, overlap_width: int) -> npt.NDArray[np.float32]:
    '''
    DEPRECATED:
    Merge two overlapping tile regions using alpha blending
    Args:
        tile0: The first tile
        tile1: The second tile
        updown: Whether the tiles are overlapping vertically or horizontally
    '''
    assert tile0.shape == tile1.shape
    if updown:
        alphas = np.linspace(0, 1, overlap_width)
        return tile0 * alphas[:, None] + tile1 * (1 - alphas)[:, None]
    else:
        alphas = np.linspace(0, 1, overlap_width)
        return tile0 * alphas[None, :] + tile1 * (1 - alphas)[None, :]
    
def get_merge_filter(shape: npt.NDArray[np.uint8], merge_mode: int, overlap_width: int, 
                     sides: int=0b1111, tile0: Optional[npt.NDArray[np.float32]]=None, 
                     tile1: Optional[npt.NDArray[np.float32]]=None) -> npt.NDArray[np.float32]:
    '''
    Get the 0-1 filter to achieve the desired merge mode
    Args:
        shape: The shape of the tile
        merge_mode: The mode to use for merging (0: none, 1: alpha blend, 2: graph cut, 3: average)
        overlap_width: The width of the overlap
        sides: Which sides to merge (0b0001: left, 0b0010: right, 0b0100: up, 0b1000: down)
        tile0: The first tile (only needed for graph cut merge)
        tile1: The second tile (only needed for graph cut merge)
    Returns:
        The filter
    '''
    to_return = np.ones(shape[:2])
    if overlap_width == 0:
        return to_return
    if merge_mode == 0:
        if sides & 0b0001:
            to_return[:, :overlap_width] = 1
        elif sides & 0b0010:
            to_return[:, -overlap_width:] = 0
        elif sides & 0b0100:
            to_return[:overlap_width, :] = 1
        elif sides & 0b1000:
            to_return[-overlap_width:, :] = 0
        return to_return
    elif merge_mode == 1:
        right = np.ones(shape[:2])
        left = np.ones(shape[:2])
        left[:, :overlap_width] = np.linspace(0, 1, overlap_width)[None, :]
        right[:, -overlap_width:] = np.linspace(1, 0, overlap_width)[None, :]
        up = np.ones(shape[:2])
        down = np.ones(shape[:2])
        up[:overlap_width, :] = np.linspace(0, 1, overlap_width)[:, None]
        down[-overlap_width:, :] = np.linspace(1, 0, overlap_width)[:, None]
        if sides & 0b0001:
            to_return *= left
        if sides & 0b0010:
            to_return *= right
        if sides & 0b0100:
            to_return *= up
        if sides & 0b1000:
            to_return *= down
        return to_return
    elif merge_mode == 2:
        if tile0 is None or tile1 is None:
            raise ValueError('tile0 and tile1 must be provided for graph cut merge')
        if sides & 0b0001:
            to_return[:, :overlap_width] = graph_cut_merge(tile0, tile1, False, sides)
        elif sides & 0b0010:
            to_return[:, -overlap_width:] = graph_cut_merge(tile0, tile1, False, sides)
        elif sides & 0b0100:
            to_return[:overlap_width, :] = graph_cut_merge(tile0, tile1, True, sides)
        elif sides & 0b1000:
            to_return[-overlap_width:, :] = graph_cut_merge(tile0, tile1, True, sides)
        return to_return
    elif merge_mode == 3:
        if sides & 0b0001:
            to_return[:, :overlap_width] = 0.5
        elif sides & 0b0010:
            to_return[:, -overlap_width:] = 0.5
        elif sides & 0b0100:
            to_return[:overlap_width, :] = 0.5
        elif sides & 0b1000:
            to_return[-overlap_width:, :] = 0.5
        return to_return
    else:
        raise ValueError(f'Invalid merge mode {merge_mode}')

LEFT = 0b0001
RIGHT = 0b0010
UP = 0b0100
DOWN = 0b1000

NONE = 0
ALPHA_BLEND = 1
GRAPH_CUT = 2
AVERAGE = 3

def add_tile(full_size: npt.NDArray[np.float32], tile: npt.NDArray[np.float32], y: int, x: int, 
             y_ow: int, x_ow: int, merge_mode: int=ALPHA_BLEND, use_poisson: bool=False) -> npt.NDArray[np.float32]:
    '''
    Add the tile to the full size image at the given location.
    Args:
        full_size: The full size image
        tile: The tile to add
        y: The y pixel coordinate of the tile
        x: The x pixel coordinate of the tile
        y_ow: The y overlap width
        x_ow: The x overlap width
        merge_mode: The mode to use for merging (0: none, 1: alpha blend, 2: graph cut, 3: average)
        use_poisson: Whether to use poisson blending
    Returns:
        The full size image with the tile added
    '''
    # TODO Check wrapping around
    tile_h = tile.shape[0] - y_ow
    tile_w = tile.shape[1] - x_ow

    to_return = full_size.copy()
    dest_tile = get_tile(to_return, y, x, tile_w+max(y_ow, x_ow))
    dest_tile = dest_tile[:tile_h+y_ow, :tile_w+x_ow]
    src_tile = tile

    src_filter = get_merge_filter(
        (tile_h+y_ow, tile_w+x_ow), merge_mode, y_ow, UP, dest_tile[:y_ow, :], src_tile[:y_ow, :]
    ) * get_merge_filter(
        (tile_h+y_ow, tile_w+x_ow), merge_mode, x_ow, LEFT, dest_tile[:, :x_ow], src_tile[:, :x_ow]
    ) * get_merge_filter(
        (tile_h+y_ow, tile_w+x_ow), merge_mode, y_ow, DOWN, dest_tile[-y_ow:, :], src_tile[-y_ow:, :]
    ) * get_merge_filter(
        (tile_h+y_ow, tile_w+x_ow), merge_mode, x_ow, RIGHT, dest_tile[:, -x_ow:], src_tile[:, -x_ow:]
    )

    # TODO Investigate this more
    # src_filter = np.zeros_like(src_filter)
    # interped = np.clip(np.linspace(0, 1, y_ow)[None, :] + np.linspace(0, 1, x_ow)[:, None], 0, 1)
    # # Set each corner of src_filter to a rotated version of interped
    # src_filter[:y_ow, :x_ow] = interped
    # src_filter[-y_ow:, :x_ow] = np.rot90(interped, 1)
    # src_filter[-y_ow:, -x_ow:] = np.rot90(interped, 2)
    # src_filter[:y_ow, -x_ow:] = np.rot90(interped, 3)

    dest_filter = np.ones_like(src_filter) - src_filter

    if use_poisson:
        mask = (src_filter>=0.5)*255
        mask = clean_mask(mask.astype(np.uint8))
        mask = np.dstack([mask, mask, mask])
        center = (dest_tile.shape[1]//2, dest_tile.shape[0]//2)
        src_tile = np.dstack([(src_tile*255).astype(np.uint8)]*3)
        dest_tile = (dest_tile*255).astype(np.uint8)
        dest_tile = np.dstack([dest_tile, dest_tile, dest_tile])
        merged = cv2.seamlessClone(src_tile, dest_tile, mask, center, cv2.MONOCHROME_TRANSFER).mean(axis=2)/255
        
    else:
        src_tile = src_tile * src_filter
        dest_tile = dest_tile * dest_filter
        merged = src_tile + dest_tile

    set_tile(to_return, merged, y, x)

    return to_return

def clean_mask(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    '''
    Clean a mask making sure there is a clean divide
    Args:
        mask: The mask to clean
    Returns:
        The cleaned mask
    '''

    to_return = mask.copy()
    to_return = cv2.erode(to_return, np.ones((3, 3), np.uint8), iterations=3)
    to_return = cv2.dilate(to_return, np.ones((3, 3), np.uint8), iterations=4)
    to_return[mask != 255] = 0
    return to_return
    

def get_display_dem(full_sized_dem: npt.NDArray[np.float32], size: int, tile_w: int=256, merge_mode: int=2, use_poisson: bool=False) -> npt.NDArray[np.float32]:
    '''
    Get the dem to display by shrinking and merging tiles
    Args:
        full_sized_dem: The full sized dem
        size: The size of the dem
        tile_w: The width of a tile
        merge_mode: The mode to use for merging (0: none, 1: alpha blend, 2: graph cut, 3: average)
    Returns:
        The dem to display
    '''

    full_sized = full_sized_dem.copy()
    if len(full_sized.shape) == 3:
        full_sized = full_sized[:, :, 0]

    max_y = 2**size # TODO Test for more than 2 y tiles
    max_x = 2**(size+1)

    height, width = tile_w * max_y, tile_w * max_x

    
    ow = (full_sized.shape[1] - tile_w * max_x) // max_x
    to_return_y = np.zeros((height, full_sized.shape[1]), np.float32)
    for max_v in (max_y-1, max_x-1):
        w = tile_w if max_v == 0 else tile_w+ow
        to_return_y[:w, :] = full_sized[:w, :]
        for y in range(0, max_v):
            to_return_y = np.roll(to_return_y, -tile_w, axis=0)
            full_sized = np.roll(full_sized, -tile_w-ow, axis=0)
            dest_tile = to_return_y[:ow, :]
            src_tile = full_sized[:ow, :]
            src_filter = get_merge_filter((tile_w+ow, full_sized.shape[1]), merge_mode, ow, 0b0100, dest_tile, src_tile)
            if use_poisson:
                w = tile_w if y == max_v-1 else tile_w+ow
                to_return_y[ow:w, :] = full_sized[ow:w, :]
                dest_tile = to_return_y[:w, :]
                src_tile = full_sized[:w, :]

                if y == max_x - 2 and max_v == max_x - 1:
                    # Merge the far end of the tile at the same time when wrapping around for the last merge
                    src_filter *= get_merge_filter((tile_w+ow, full_sized.shape[1]), merge_mode, ow, 0b1000, dest_tile[-ow:, :], src_tile[-ow:, :])

                mask = (src_filter[:w, :]>=0.5)*255
                mask = clean_mask(mask.astype(np.uint8))
                mask = np.dstack([mask, mask, mask])
                center = (dest_tile.shape[1]//2, dest_tile.shape[0]//2)
                src_tile = np.dstack([(src_tile*255).astype(np.uint8)]*3)
                dest_tile = (dest_tile*255).astype(np.uint8)
                dest_tile = np.dstack([dest_tile, dest_tile, dest_tile])
                merged = cv2.seamlessClone(src_tile, dest_tile, mask, center, cv2.MONOCHROME_TRANSFER).mean(axis=2)/255
                to_return_y[:w, :] = merged
            else:
                dest_filter = np.ones_like(src_filter) 
                dest_filter[-ow:, :] = 1 - src_filter[:ow, :] #get_merge_filter((tile_w+ow, full_sized.shape[1]), merge_mode, ow, 0b1000, dest_tile, src_tile)
                src_tile = full_sized[:tile_w+ow, :] * src_filter[:tile_w+ow, :]
                dest_tile = to_return_y[tile_w:tile_w+ow, :]
                src_tile[:ow, :] += to_return_y[:ow, :] * dest_filter[-ow:, :]
                if y == max_x - 2 and max_v == max_x - 1:
                    # Merge the far end of the tile at the same time when wrapping around for the last merge
                    src_filter = get_merge_filter((tile_w+ow, full_sized.shape[1]), merge_mode, ow, 0b1000, dest_tile, src_tile[-ow:, :])
                    dest_filter = np.ones_like(src_filter)
                    dest_filter[:ow, :] = 1 - src_filter[-ow:, :]
                    src_tile[-ow:, :] *= src_filter[-ow:, :]
                    src_tile[-ow:, :] += dest_tile * dest_filter[:ow, :]
                to_return_y[:tile_w+ow, :] = src_tile
        if max_v != max_x:
            to_return_y = np.roll(to_return_y, -tile_w, axis=0)
        to_return_y = np.transpose(to_return_y, (1, 0))
        full_sized = to_return_y.copy()
        to_return_y = np.zeros((width, height), np.float32)
    if len(full_sized_dem.shape) == 3:
        full_sized = np.expand_dims(full_sized, axis=2)
    return full_sized

def test_image(size: int, ow: int, tile_w: int):
    '''
    Create a test image
    Args:
        size: The size of the planetCFG
        ow: The overlap width
        tile_w: The width of a tile

    '''
    y_tiles = 2**size
    x_tiles = 2**(size+1)
    planet_cfg = PlanetConfig(size, image_mode='none')
    to_return = np.zeros((tile_w*y_tiles+ow*(y_tiles-1), tile_w*x_tiles+ow*x_tiles), np.float32)
    for y in range(y_tiles):
        for x in range(x_tiles):
            _, dem = gen_tile_pair(0, y*256, x*256, planet_cfg)
            dem = cv2.resize(dem, (tile_w, tile_w))/255
            to_return[y*(tile_w+ow):y*(tile_w+ow)+tile_w, x*(tile_w+ow):x*(tile_w+ow)+tile_w] = dem
            if y > 0:
                to_return[y*(tile_w+ow)-ow:y*(tile_w+ow), x*(tile_w+ow):x*(tile_w+ow)+tile_w] = dem[:ow, :]
            if x == 0:
                to_return[y*(tile_w+ow):y*(tile_w+ow)+tile_w, -ow:] = dem[:, :ow]
                if y > 0:
                    to_return[y*(tile_w+ow)-ow:y*(tile_w+ow), -ow:] = dem[:ow, :ow]
            else:
                to_return[y*(tile_w+ow):y*(tile_w+ow)+tile_w, x*(tile_w+ow)-ow:x*(tile_w+ow)] = dem[:, :ow]
                if y > 0:
                    to_return[y*(tile_w+ow)-ow:y*(tile_w+ow), x*(tile_w+ow)-ow:x*(tile_w+ow)] = dem[:ow, :ow]
    return to_return

def generate_overlap_test():
    full_dem = np.array(img.open('1024x512_dem.png')).astype(np.float32) / 255
    dem = full_dem[:, :, 0]
    x_overlap = full_dem[:, :, 1]
    y_overlap = full_dem[:, :, 2]
    # z_overlap = full_dem[:, :, 3]
    to_return = dem.copy()
    y_tiles = 2
    x_tiles = 4
    ow = 128
    for y in range(y_tiles):
        for x in range(x_tiles):
            src_y = y * 256
            src_x = (x+1) * 256
            src_tile = get_tile(x_overlap, src_y, src_x-128)
            to_return = add_tile(
                to_return, 
                src_tile, 
                src_y,
                src_x - 128,
                0,
                ow,
                ALPHA_BLEND,
                False
            )
    for y in range(y_tiles-1):
        for x in range(x_tiles):
            src_y = (y+1) * 256
            src_x = x * 256
            src_tile = get_tile(y_overlap, src_y-128, src_x)
            to_return = add_tile(
                to_return, 
                src_tile, 
                src_y - 128,
                src_x,
                ow,
                0,
                ALPHA_BLEND,
                False
            )
    img.fromarray((to_return * 255).astype(np.uint8)).show()
    return to_return

if __name__ == '__main__':

    size = 1
    tile_w = 256
    ow = 64
    images = []
    texts = []
    merge_modes = [
        'None',
        'Alpha Blend',
        'Graph Cut',
        'Average',
    ]

    generate_overlap_test()

    for merge_mode in range(0, 4):
        for use_poisson in [True, False]:
            full_sized = np.array(img.open('1280x576.png')).astype(np.float32) / 255
            test = test_image(size, ow, tile_w)
            t1 = time()
            full_sized = get_display_dem(full_sized, size, tile_w, merge_mode=merge_mode, use_poisson=use_poisson)
            # Uncomment this line to see results of add_tile
            # full_sized = add_tile(full_sized, test[:tile_w+ow, :tile_w+ow], 0, 0, tile_w, merge_mode=merge_mode, use_poisson=use_poisson)
            images.append(img.fromarray((full_sized * 255).astype(np.uint8)))
            texts.append(f"{time()-t1:.3f}s")
        
    grid = image_grid(images, len(merge_modes), 2, texts, col_labels=["Poisson: True", "Poisson: False"], row_labels=merge_modes)
    grid.save('Test_Merge_Modes.png')
