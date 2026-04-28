from math import ceil, log2
import numpy as np
import itertools
import random
from time import time


class QuadTreeLevel:
    children: list["QuadTreeLevel"]
    complete: bool
    num_levels: int

    def __init__(self, num_levels: int, y: int = 0, x: int = 0):
        self.complete = False
        if num_levels <= 0:
            self.children = []
            # Leaf nodes have coords
            self.y = y
            self.x = x
        else:
            half = 2 ** (num_levels - 1)
            self.children = [QuadTreeLevel(num_levels - 1, y + (i // 2) * half, x + (i % 2) * half) for i in range(4)]
            self.y = None
            self.x = None
        self.num_levels = num_levels

    def set_complete(self, y: int, x: int) -> None:
        if self.num_levels <= 0:
            assert y == x == 0, "Expected last recursion to be the correct coord"
            self.complete = True
            return

        half = 2 ** (self.num_levels - 1)
        child_to_set = 0
        if y >= half:
            child_to_set += 2
            y -= half
        if x >= half:
            child_to_set += 1
            x -= half
        self.children[child_to_set].set_complete(y, x)
        self.complete = all([child.complete for child in self.children])

    def is_complete(self, y: int, x: int) -> bool:
        if self.complete:
            return True
        if len(self.children) == 0:
            return self.complete
        half = 2 ** (self.num_levels - 1)
        child_to_get = 0
        if y >= half:
            child_to_get += 2
            y -= half
        if x >= half:
            child_to_get += 1
            x -= half
        return self.children[child_to_get].is_complete(y, x)

    def __str__(self) -> str:
        return f"{self.complete}\n -" + "\n -".join([str(child) for child in self.children])

    def get_random_incomplete(self) -> tuple[int, int] | tuple[None, None]:
        if self.complete:
            return None, None
        if len(self.children) == 0:
            return self.y, self.x
        incomplete_children = []
        for i, child in enumerate(self.children):
            if not child.complete:
                incomplete_children.append(i)
        if len(incomplete_children) == 0:
            raise Exception("Current node not marked as complete but all children are complete. Something is wrong.")
        random_i: int = random.choice(incomplete_children)
        return self.children[random_i].get_random_incomplete()


class QuadTreeTiles:
    def __init__(self, mask: np.ndarray, num_levels: int = 3) -> None:
        assert len(mask.shape) == 2, "Mask must be 2D"
        assert mask.dtype == bool, "Only boolean masks are permitted"
        self.mask = mask
        self.num_levels = num_levels
        self.init_levels()
        self.init_tiles()

    def init_levels(self):
        """Init the quad tree levels"""
        self.levels = QuadTreeLevel(self.num_levels)

    def init_tiles(self):
        self.mask_h, self.mask_w = self.mask.shape
        self.num_tiles = 2 ** self.num_levels
        self.tile_h = int(ceil(self.mask_h / self.num_tiles))
        self.tile_w = int(ceil(self.mask_w / self.num_tiles))
        self.tiles = []
        for (y_tile, x_tile) in itertools.product(range(self.num_tiles), range(self.num_tiles)):
            y_start = y_tile * self.tile_h
            x_start = x_tile * self.tile_w
            y_end = min(y_start + self.tile_h, self.mask_h)
            x_end = min(x_start + self.tile_w, self.mask_w)
            self.tiles.append(self.mask[y_start:y_end, x_start:x_end])

    def get_false_coord(self) -> tuple[int, int] | tuple[None, None]:
        """Get a random coordinate that is set to False"""
        y, x = self.levels.get_random_incomplete()
        if y is None or x is None:
            return None, None
        # print(y, x)
        tile_i = y * self.num_tiles + x
        tile = self.tiles[tile_i]
        incomplete = np.where(~tile)
        num_incomplete = len(incomplete[0])
        assert num_incomplete > 0, "This tile is supposed to be incomplete since it is incomplete in the level."
        random_i = random.randint(0, num_incomplete - 1)
        y_off = incomplete[0][random_i]
        x_off = incomplete[1][random_i]
        return y * self.tile_h + y_off, x * self.tile_w + x_off

    def add_tile(self, ys: np.ndarray, xs: np.ndarray) -> None:
        """Set a tile of coordinates to True"""
        ys = ys.flatten()
        xs = xs.flatten()
        y_blocks = ys // self.tile_h
        x_blocks = xs // self.tile_w
        tile_indices = y_blocks * self.num_tiles + x_blocks
        unique_indices = np.unique(tile_indices)
        for i in unique_indices:
            y = i // self.num_tiles
            x = i % self.num_tiles
            if self.levels.is_complete(y, x):
                continue
            tile = self.tiles[i]
            y_rems = ys[(tile_indices == i)] % self.tile_h
            x_rems = xs[(tile_indices == i)] % self.tile_w
            tile[y_rems, x_rems] = True
            self.tiles[i] = tile
            if np.all(tile):
                self.levels.set_complete(y, x)

    def all_complete(self) -> bool:
        return self.levels.complete


class QuadTreeMaskWrapper:
    def __init__(self, mask: np.ndarray, num_levels: int | None = 7) -> None:
        assert len(mask.shape) == 2, "Mask must be 2D"
        assert mask.dtype == bool, "Only boolean masks are permitted"
        self.mask = mask
        self.num_levels = self.auto_num_levels(mask.shape) if num_levels is None else num_levels
        self.init_levels()
        self.init_tiles()

    def auto_num_levels(self, mask_shape: tuple[int, int]) -> int:
        h, w = mask_shape
        h_tiles = ceil(h / 128)
        w_tiles = ceil(w / 128)
        h_levels = ceil(log2(h_tiles))
        w_levels = ceil(log2(w_tiles))
        return max(h_levels, w_levels, 2)

    def init_levels(self):
        """Init the quad tree levels"""
        self.levels = QuadTreeLevel(self.num_levels)

    def init_tiles(self):
        t1 = time()
        self.mask_h, self.mask_w = self.mask.shape
        self.num_tiles = 2 ** self.num_levels
        self.tile_h = int(ceil(self.mask_h / self.num_tiles))
        self.tile_w = int(ceil(self.mask_w / self.num_tiles))

        if np.any(self.mask):
            for y, x in itertools.product(range(self.num_tiles), range(self.num_tiles)):
                start_y = y * self.tile_h
                start_x = x * self.tile_w
                end_y = min(start_y + self.tile_h, self.mask_h - 1)
                end_x = min(start_x + self.tile_w, self.mask_w - 1)
                tile = self.mask[start_y:end_y, start_x:end_x]
                if np.all(tile):
                    self.levels.set_complete(y, x)
        t2 = time()
        if t2 - t1 > 0.1:
            print(
                f"Initialised tiles for {self.mask_w}x{self.mask_h} mask",
                f"with {self.num_tiles} {self.tile_w}x{self.tile_h} tiles",
                f"in {t2 - t1:.3f}s"
            )

    def get_false_coord(self) -> tuple[int, int] | tuple[None, None]:
        """Get a random coordinate that is set to False"""
        while not self.levels.complete:
            y, x = self.levels.get_random_incomplete()
            if y is None or x is None:
                return None, None
            # print(y, x)
            start_y = y * self.tile_h
            start_x = x * self.tile_w
            end_y = min(start_y + self.tile_h, self.mask_h)
            end_x = min(start_x + self.tile_w, self.mask_w)

            tile = self.mask[start_y:end_y, start_x:end_x]
            incomplete = np.where(~tile)
            num_incomplete = len(incomplete[0])
            if num_incomplete == 0:
                self.levels.set_complete(y, x)
                continue
            random_i = random.randint(0, num_incomplete - 1)
            y_off = incomplete[0][random_i]
            x_off = incomplete[1][random_i]
            return start_y + y_off, start_x + x_off
        return None, None

    def add_tile(self, ys: np.ndarray, xs: np.ndarray) -> None:
        """Set a tile of coordinates to True"""
        self.mask[ys, xs] = True

    def all_complete(self) -> bool:
        return self.levels.complete


class RowMask:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        h, w = mask.shape
        self.sections = h
        self.H = h
        self.section_done = np.zeros(self.sections, dtype=bool)

    def get_false_coord(self) -> tuple[int, int] | tuple[None, None]:
        valid_tile = False
        while not valid_tile:
            sections = np.where(self.section_done == 0)[0]
            if len(sections) == 0:
                return None, None
            else:
                sec = np.random.choice(sections)
                u_min = int(round(sec * self.H / self.sections))
                u_max = int(round((sec + 1) * self.H / self.sections)) if sec != self.sections - 1 else self.H

                u, v = np.where(self.mask[u_min:u_max] == 0)
                if len(u) == 0:
                    self.section_done[sec] = True
                    continue
                u += u_min
            u = u.flatten()
            v = v.flatten()
            # np.random.seed(self.seed + self.t)
            idx = np.random.choice(len(u))
            u_i, v_i = u[idx], v[idx]
            return u_i, v_i

    def add_tile(self, ys: np.ndarray, xs: np.ndarray) -> None:
        self.mask[ys, xs] = True

    def all_complete(self) -> bool:
        return np.all(self.mask)


def benchmark_masks():
    tile_size = 256
    # h, w = 1304, 7824
    # h, w = 2608, 15648
    h, w = 5216, 31296
    num_tests = 3
    mask = np.zeros((h, w), dtype=bool)
    times = {}
    for n in range(num_tests):
        row_mask = RowMask(mask.copy())
        ds_types = {f"quad_{i}": QuadTreeMaskWrapper(mask.copy(), num_levels=i) for i in range(6, 9)}
        ds_types.update({"row": row_mask})
        for i, key in enumerate(ds_types):
            ds: QuadTreeMaskWrapper | RowMask = ds_types[key]
            print(key, f"{100 * (i + 1) / len(ds_types):.1f}%")
            coord_key = key + "_coord"
            tile_key = key + "_tile"
            while True:
                t1 = time()
                y, x = ds.get_false_coord()
                coord_time = time() - t1
                times.setdefault(coord_key, [])
                times[coord_key].append(coord_time)
                if y is None or x is None:
                    if not ds.all_complete():
                        print("Not all tiles complete for", key)
                    break
                ys = np.arange(tile_size)
                xs = np.arange(tile_size)
                xs, ys = np.meshgrid(xs, ys)
                ys += y
                xs += x
                ys = np.minimum(ys, h - 1)
                xs = np.minimum(xs, w - 1)
                assert ys.max() < h
                assert xs.max() < w
                t1 = time()
                ds.add_tile(ys, xs)
                tile_time = time() - t1
                times.setdefault(tile_key, [])
                times[tile_key].append(tile_time)
            print(coord_key, f"{sum(times.get(coord_key, [0])):.3f}")
            print(tile_key, f"{sum(times.get(tile_key, [0])):.3f}")
    print("Key", "Total", "Max", "Min", "Mean", "Length")
    for key, val in times.items():
        print(
            key,
            f"{sum(val):.3f}s",
            f"{max(val) * 1000:.3f}ms",
            f"{min(val) * 1000:.3f}ms",
            f"{sum(val) / len(val) * 1000:.3f}ms",
            len(val),
            sep="\t"
        )


def main():
    num_levels = 3
    num_tiles = 2 ** num_levels
    levels = QuadTreeLevel(num_levels)
    for i, (y, x) in enumerate(itertools.product(range(num_tiles), range(num_tiles))):
        levels.set_complete(y, x)
        pcnt = (i + 1) / (num_tiles ** 2)
        print(pcnt, [lev.complete for lev in levels.children], levels.get_random_incomplete())
    h, w = 405, 996
    mask = np.zeros((h, w), dtype=bool)  # Test with some arbitrary coords
    quad_mask = QuadTreeTiles(mask, num_levels=num_levels)
    max_y, max_x = 0, 0
    for i in range(10000):
        y, x = quad_mask.get_false_coord()
        # print(y, x)
        max_y = max(y, max_y)
        max_x = max(x, max_x)
        assert y < h and x < w and y >= 0 and x >= 0
    print(max_y, max_x)
    return


if __name__ == "__main__":
    # main()
    benchmark_masks()
