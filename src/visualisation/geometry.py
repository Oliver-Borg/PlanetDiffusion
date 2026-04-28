from PIL import Image

import numpy as np
import pyrr
from OpenGL.GL import *
import pygame as pg

from ..core.derivative import SGF_to_gradient, elevation_to_gradient, gradient_to_elevation, image_to_SGF
from ..core.elevation import get_elevation_data_of_tile
from ..core.utils import load_image


def generate_faces(shape):
    num_vertices_y, num_vertices_x = shape

    a = np.arange(num_vertices_x * (num_vertices_y - 1) - 1)
    b = a + 1
    c = b + (num_vertices_x - 1)
    d = c + 1

    first = np.column_stack([
        a,
        c,
        b
    ])

    second = np.column_stack([
        b,
        c,
        d
    ])

    sl = slice(num_vertices_x - 1, None, num_vertices_x)
    first = np.delete(first, sl, axis=0)
    second = np.delete(second, sl, axis=0)

    return np.concatenate((first, second)).astype(np.uint32)


def calculate_normals(gradient):
    dh_dx, dh_dy = gradient

    temp_shape = (*dh_dx.shape, 2)
    u_arr = np.zeros(temp_shape)
    u_arr[:, :, 0] = 1
    u = np.dstack((u_arr, dh_dx))

    v_arr = np.zeros(temp_shape)
    v_arr[:, :, 1] = 1
    v = np.dstack((v_arr, dh_dy))

    return np.cross(u, v)


class Material:
    """Used to create material objects from MTL files

    Adapted from GetIntoGameDev's tutorial: https://www.youtube.com/watch?v=ZK1WyCMK12E
    """

    def __init__(self, dict_of_textures, ambient=0.2, shine=8, kd=1, ks=1):
        """Create a Material object from a file"""

        self.ambient = ambient
        self.shine = shine
        self.kd = kd
        self.ks = ks

        self.textures = {}

        # Load image
        for texture_type, texture_file in dict_of_textures.items():
            self.add_texture(texture_type, texture_file)

    def add_texture(self, texture_type, texture_file):
        # Generate a texture name for this object
        new_texture = glGenTextures(1)
        self.textures[texture_type] = new_texture
        glBindTexture(GL_TEXTURE_2D, new_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load image
        if isinstance(texture_file, Image.Image):
            # https://stackoverflow.com/a/64182629/13989043
            # https://gist.github.com/jawa0/4003034
            image = pg.image.fromstring(
                texture_file.tobytes(), texture_file.size, texture_file.mode)
        else:  # is a path
            image = pg.image.load(texture_file)

        image_width, image_height = image.get_rect().size
        img_data = pg.image.tostring(image.convert_alpha(), 'RGBA', True)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width,
                     image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def activate(self):
        for i, texture in enumerate(self.textures.values()):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, texture)

    def cleanup(self):
        glDeleteTextures(len(self.textures), list(self.textures.values()))


class Mesh:
    """Used to create mesh objects from gradient map image"""

    def __init__(self, path, res, type='elevation', normalisation_factor=25):
        """Create a Mesh object from a file"""

        if type == 'elevation':
            # TODO fix normalisation factor
            elevation = load_image(path)
            elevation *= normalisation_factor
            gradient = elevation_to_gradient(elevation, res, res)

        elif type == 'derivative':
            # Read image and convert to gradient field
            raise NotImplementedError
            # concat_image_tiles()
            # img = Image.open(path)
            # sgf = image_to_SGF(img)
            # gradient = SGF_to_gradient(sgf)

            # # Generate elevation map
            # elevation = gradient_to_elevation(
            #     gradient, res, res)
        else:
            raise

        elevation = elevation[:-1, :-1]  # Ignore last row and column
        elevation -= (np.min(elevation) + np.max(elevation)) / 2  # Center

        # Normalise scale (fit on screen)
        scale_factor = np.max(elevation.shape)
        elevation /= scale_factor

        # Calculate normal vectors
        terrain_normals = calculate_normals(gradient)

        # https://stackoverflow.com/a/44230705/13989043
        orig_indices = np.indices(elevation.shape).transpose(
            1, 2, 0).astype(np.float32)
        orig_indices = np.flip(orig_indices, axis=2)

        # Scale texture coords to be square image
        texture_coords = np.copy(orig_indices)
        texture_coords[:, :, 0] /= np.max(texture_coords[:, :, 0])
        texture_coords[:, :, 1] /= np.max(texture_coords[:, :, 1])
        texture_coords[:, :, 1] = 1 - texture_coords[:, :, 1]

        orig_indices /= scale_factor  # Fit on screen
        orig_indices -= 0.5  # Center on screen

        vertices = np.dstack(
            (orig_indices[:, :, 0], elevation, orig_indices[:, :, 1], texture_coords, terrain_normals))
        num_attribs = vertices.shape[-1]

        flattened_vertices = vertices.flatten().astype(np.float32)

        faces = generate_faces(elevation.shape)
        flattened_indices = faces.flatten().astype(np.uint32)

        self.num_indices = len(flattened_indices)

        # Create a Vertex Array Object for the mesh
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create Vertex Buffer Object for mesh
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, flattened_vertices.nbytes,
                     flattened_vertices, GL_STATIC_DRAW)

        # Create Element Buffer Object for mesh
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, flattened_indices.nbytes,
                     flattened_indices, GL_STATIC_DRAW)

        # Create Vertex Attributes Pointers here
        # Note that you will need to use ctypes.c_void_p(i) to specify the starting index
        # when using glVertexAttribPointer

        # Set stride based on the number of vertex attributes * bytes per float
        dtype = np.dtype(np.float32)
        stride = num_attribs * dtype.itemsize

        # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(0))

        # Texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(12))

        # Normals
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(20))

    def cleanup(self):
        """Clean up the Mesh"""
        glDeleteBuffers(1, (self.vbo,))
        glDeleteBuffers(1, (self.ebo,))
        glDeleteVertexArrays(1, (self.vao,))


class Object:
    """
    Used to store an Object that contains mesh, material, position,
    rotation, scale, colour, children, and visibility information
    """

    def __init__(self, mesh: Mesh, material: Material, position, eulers, scale, colour=None, children=None, hidden=False, fixed=False):
        """Create a new Object, to be placed in a scene"""
        if children is None:
            children = []

        self.mesh = mesh
        self.material = material

        self._position = np.array(position, dtype=np.float32)
        self._eulers = np.array(eulers, dtype=np.float32)
        self._scale = np.array(scale, dtype=np.float32)
        self._colour = colour
        self._children = children
        self._hidden = hidden
        self._fixed = fixed
        self.reset()

    def reset(self):
        """Reset the Object to it's original state"""
        self.position = np.copy(self._position)
        self.eulers = np.copy(self._eulers)
        self.scale = np.copy(self._scale)
        self.colour = self._colour
        self.children = self._children
        self.hidden = self._hidden
        self.fixed = self._fixed

        # Reset children
        for child in self.children:
            child.reset()

    def add_child(self, child: 'Object'):
        """Add a child to this object"""
        self.children.append(child)

    def cleanup(self):
        """Clean up this object"""
        # Clean this object's Mesh
        self.mesh.cleanup()
        self.material.cleanup()

        # Call cleanup for children
        for child in self.children:
            child.cleanup()

    def generate_transform_matrix(self):
        """Generate transformation matrix for this object, using its scale, rotation, and translation attributes"""

        # 1. Start with identity matrix
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        # 2. Scale
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_scale(
                scale=self.scale, dtype=np.float32
            )
        )

        # 3. Rotation
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_eulers(
                eulers=np.radians(self.eulers), dtype=np.float32
            )
        )

        # 4. Translation
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.position), dtype=np.float32
            )
        )

        return model_transform


class Light:
    def __init__(self, colour, strength, obj=None, position=None) -> None:

        self._position = np.array(position, dtype=np.float32)

        # assume ONE colour per light (so ambient, diffuse
        # and specular light colour is all the same)
        self.colour = np.array(colour)/255  # normalised

        self.strength = strength

        self.object = obj

        # RGB intensity
        # self.ambprod = ambprod
        # self.diffprod = diffprod
        # self.specprod = specprod

    @property
    def position(self):
        if self.object is not None:
            return self.object.position

        return self._position

    @position.setter
    def position(self, new_position):
        if self.object is not None:
            self.object.position = new_position
        else:
            self._position = new_position

    def rotate_around_axis(self, angle, axis):
        # Generate rotation matrix
        eulers = [0, 0, 0]
        eulers[axis] = np.radians(angle)
        mat = pyrr.matrix33.create_from_eulers(
            eulers=eulers, dtype=np.float32
        )

        # Apply rotation to camera's position and up vector
        self.position = pyrr.matrix33.apply_to_vector(
            mat=mat,
            vec=self.position
        )
