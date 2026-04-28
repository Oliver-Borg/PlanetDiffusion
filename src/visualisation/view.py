import argparse
import math
import os
from enum import Enum, IntEnum

import numpy as np
import pyrr
import pygame as pg
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GL import *
from PIL import Image

from .geometry import Material, Mesh, Object, Light
from ..core.constants import LODS, MAX_ELEVATION_LEVEL

# Define constant colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SUNLIGHT = (255, 255, 192)


class Mode(Enum):
    """Used to specify modes"""
    SCALE_OBJECT = 0
    ROTATE_OBJECT = 1
    ROTATE_CAMERA = 2
    ROTATE_LIGHTS = 3
    CHANGE_AXIS = 4


class Axes(IntEnum):
    """Enumerate X, Y, and Z axes"""
    X = 0
    Y = 1
    Z = 2


class State:
    """Used to store the current state of the program"""

    def __init__(self):
        """Create a new State object"""
        self.reset()

    def reset(self):
        """Reset the state to its defaults"""
        self.mode = Mode.SCALE_OBJECT
        self.axis = Axes.Y


def loadShaderProgram(vertex, fragment):
    """Load a shader program, given it's vertex and fragment file"""
    with open(vertex) as f:
        vertex_src = f.readlines()

    with open(fragment) as f:
        fragment_src = f.readlines()

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))

    return shader


class Camera:

    def __init__(self, position, target, up):
        # Position camera slightly away

        self._position = position
        self._target = target
        self._up = up

        self.reset()

    def reset(self):
        """Reset the Object to it's original state"""
        self.position = np.copy(self._position)
        self.target = np.copy(self._target)
        self.up = np.copy(self._up)

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
        self.up = pyrr.matrix33.apply_to_vector(
            mat=mat,
            vec=self.up
        )


class Scene:
    def __init__(self, camera: Camera):
        self.camera = camera
        self.objects = []
        self.lights = []
        self.skybox = None

    def add_lights(self, *lights: Light):
        self.lights.extend(*lights)

    def add_objects(self, *objs: Object):
        self.objects.extend(*objs)

    def reset(self, remove_objects=False, remove_lights=False):
        self.camera.reset()

        for obj in self.objects:
            obj.reset()

        if remove_objects:
            self.objects.clear()

        if remove_lights:
            self.lights.clear()

    def cleanup(self):
        for obj in self.objects:
            obj.cleanup()


class Application:
    """Class used for the base application"""

    def __init__(self,
                 window_width, window_height,               # Window
                 fovy, near, far,                           # View
                 vertex_shader_path, fragment_shader_path,  # Shader

                 elevation_image, satellite_image, factor                # Tile to display
                 ):
        """Create a new Application object"""
        self.window_width = window_width
        self.window_height = window_height
        self.fovy = fovy
        self.near = near
        self.far = far

        self.elevation_image = elevation_image
        self.satellite_image = satellite_image
        self.factor = factor

        # Set up pygame
        pg.init()

        pg.display.gl_set_attribute(
            pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 2)

        pg.display.set_caption('Terrain viewer')
        pg.display.set_mode((window_width, window_height),
                            pg.OPENGL | pg.DOUBLEBUF)

        self.clock = pg.time.Clock()

        # Set up OpenGL
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(*WHITE, 1)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        vertex_shader_path = os.path.join(dir_path, vertex_shader_path)
        fragment_shader_path = os.path.join(dir_path, fragment_shader_path)

        self.shader = loadShaderProgram(
            vertex_shader_path, fragment_shader_path)
        glUseProgram(self.shader)

        rot = np.pi / 12
        rad = 0.75

        pi_over_2 = np.pi/2
        rot = rot - pi_over_2

        # Create default scene
        self.scene = Scene(
            # Manage scene's camera and its movements
            Camera(
                [0, rad * math.cos(rot), rad * math.sin(rot)],
                [0, 0, 0],  # Look at origin
                [0, rad * math.cos(pi_over_2 + rot), rad * \
                 math.sin(pi_over_2 + rot)],
                # [0, math.sin(rot), -math.cos(rot)]
            )
        )

        # Used to keep track of state
        self.current_state = State()
        print('[START] Mode =', self.current_state.mode.name,
              'and Axis =', self.current_state.axis.name)

        # Set texture and lighting defaults
        # Always set imageTexture to be 1st texture and normalMap to be 2nd texture
        glUniform1i(
            glGetUniformLocation(self.shader, 'mat.imageTexture'),
            0
        )
        glUniform1i(
            glGetUniformLocation(self.shader, 'mat.normalMap'),
            1
        )

        # Set up view matrix (usually only done once)
        self.update_view_matrix()

        # Set up projection matrix (usually only done once)
        self.update_projection_matrix()

    def update_projection_matrix(self):
        """Update the projection matrix, which projects to the window"""
        glUseProgram(self.shader)

        projection_matrix = pyrr.matrix44.create_perspective_projection(
            fovy=self.fovy, aspect=self.window_width/self.window_height,
            near=self.near, far=self.far, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, 'projection'),
            1, GL_FALSE, projection_matrix
        )

    def update_view_matrix(self):
        """Update the view matrix, which projects to the camera/view space"""
        glUseProgram(self.shader)

        view_matrix = pyrr.matrix44.create_look_at(
            self.scene.camera.position,
            self.scene.camera.target,
            self.scene.camera.up
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, 'view'),
            1, GL_FALSE, view_matrix
        )

    def reset(self):
        """Reset the application to its original state"""
        self.current_state.reset()
        self.scene.reset()
        self.update_projection_matrix()
        self.update_view_matrix()

    def setup_scene(self):
        # Set up scene

        res = LODS[MAX_ELEVATION_LEVEL - self.factor][0]

        sat = Image.open(self.satellite_image)
        material = Material({
            'texture': sat,
        })

        # Create root object
        object_mesh = Mesh(self.elevation_image, res)
        main_object = Object(mesh=object_mesh,
                             material=material,
                             position=[0, 0, 0],  # Centered
                             eulers=[0, 0, 0],    # No rotation
                             scale=[0.5, 0.5, 0.5],
                             )

        rot = math.pi/4
        rad = 2
        self.scene.lights.append(
            Light(SUNLIGHT, 1, position=[
                rad * math.cos(rot),
                2,
                rad * math.sin(rot),
            ])
        )

        self.scene.add_objects([main_object])

    def run(self):
        """Start the main loop"""

        self.setup_scene()

        # Adjust to more reasonable multipliers/factors
        SCALE_MULTIPLIER = 0.01         # In units
        ANGLE_MULTIPLIER = 1            # In degrees

        # Used to handle key presses while in different modes,
        # where changes made to the object are continuous
        MODE_HANDLER = {
            # Mode : (attribute to change, change multiplier)
            Mode.SCALE_OBJECT: ('scale', SCALE_MULTIPLIER),
            Mode.ROTATE_OBJECT: ('eulers', ANGLE_MULTIPLIER),
        }

        # Used to switch modes, based on key press
        KEY_MODE_MAPPER = {
            # Key: mode to change to on press
            pg.K_s: Mode.SCALE_OBJECT,
            pg.K_r: Mode.ROTATE_OBJECT,
            pg.K_c: Mode.ROTATE_CAMERA,
            pg.K_l: Mode.ROTATE_LIGHTS,
            pg.K_a: Mode.CHANGE_AXIS,
        }

        # Start the game loop
        running = True
        while running:
            # Use incoming events to change state
            for event in pg.event.get():  # Grab all of the input events detected by PyGame

                if event.type == pg.QUIT:  # This event triggers when the window is closed
                    running = False

                elif event.type == pg.KEYDOWN:

                    if event.key == pg.K_q:  # This event triggers when the q key is pressed down
                        running = False

                    elif event.key in KEY_MODE_MAPPER:  # Change mode
                        self.current_state.mode = KEY_MODE_MAPPER[event.key]
                        print('[MODE] Set mode to',
                              KEY_MODE_MAPPER[event.key].name)

                    elif event.key == pg.K_x:  # Reset the application
                        self.reset()
                        print('[EVENT] Reset application')

                    # Handle discrete changes (colour and axis)
                    if event.key in (pg.K_UP, pg.K_DOWN):
                        step = 1 if event.key == pg.K_UP else -1

                        if self.current_state.mode == Mode.CHANGE_AXIS:
                            self.current_state.axis = Axes((
                                self.current_state.axis + step) % len(Axes))
                            print('[VALUE] Set axis to',
                                  self.current_state.axis.name)

            if not running:
                break  # Some event caused the app to stop running, no need to continue

            # Handle controls to update object and camera
            keys = pg.key.get_pressed()  # checking pressed keys

            # Handle similar functionality
            if self.current_state.mode in MODE_HANDLER:
                attr, m = MODE_HANDLER[self.current_state.mode]

                for obj in self.scene.objects:
                    if obj.fixed:
                        continue
                    arr = getattr(obj, attr)

                    for key in (pg.K_UP, pg.K_DOWN):
                        if keys[key]:
                            sign = 1 if key == pg.K_UP else -1
                            arr[self.current_state.axis] += sign*m

            elif self.current_state.mode in (Mode.ROTATE_CAMERA, Mode.ROTATE_LIGHTS):
                amount = 0
                for key in (pg.K_UP, pg.K_DOWN):
                    if keys[key]:
                        amount += 1 if key == pg.K_UP else -1

                if self.current_state.mode == Mode.ROTATE_CAMERA:
                    self.scene.camera.rotate_around_axis(
                        amount*ANGLE_MULTIPLIER, self.current_state.axis)
                    self.update_view_matrix()

                else:  # ROTATE_LIGHTS
                    for light in self.scene.lights:
                        light.rotate_around_axis(
                            amount*ANGLE_MULTIPLIER, self.current_state.axis)

            # Finally, render the frame
            self.render()

        self.quit()

    def draw(self, obj: Object, parent_transform=None):
        """
        Apply transformations for an object and draw it to the screen,
        and optionally apply a parent's transform
        """
        if obj.hidden:
            return

        object_transform = obj.generate_transform_matrix()

        if parent_transform is not None:
            # Do additional transformation using the parent's transformation matrix
            # This makes the object a child of the main object, meaning that all transforms
            # applied to the parent, will be applied to the child
            object_transform = pyrr.matrix44.multiply(
                m1=object_transform,
                m2=parent_transform
            )

        # Apply model's transformation before drawing
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, 'model'),
            1, GL_FALSE, object_transform
        )

        # Activate object material
        obj.material.activate()

        # Set material ambient, shine, kd and ks values
        glUniform1f(
            glGetUniformLocation(self.shader, 'mat.ambient'),
            obj.material.ambient
        )

        glUniform1f(
            glGetUniformLocation(self.shader, 'mat.shine'),
            obj.material.shine
        )

        glUniform1f(
            glGetUniformLocation(self.shader, 'mat.kd'),
            obj.material.kd
        )
        glUniform1f(
            glGetUniformLocation(self.shader, 'mat.ks'),
            obj.material.ks
        )

        # Draw mesh
        glBindVertexArray(obj.mesh.vao)
        glDrawElements(GL_TRIANGLES, obj.mesh.num_indices,
                       GL_UNSIGNED_INT, None)

        # Draw children
        for child in obj.children:
            self.draw(child, object_transform)

    def render(self):
        """Render the next frame"""

        # Reset the frame to be clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Ensure we are using the shader
        glUseProgram(self.shader)

        # Update camera information
        glUniform3f(
            glGetUniformLocation(self.shader, 'cameraPosition'),
            *self.scene.camera.position
        )

        # Update light information
        for i, light in enumerate(self.scene.lights):
            l_name = f'lights[{i}].'
            glUniform3f(
                glGetUniformLocation(self.shader, l_name + 'position'),
                *light.position
            )

            glUniform3f(
                glGetUniformLocation(self.shader, l_name + 'colour'),
                *light.colour  # NOTE: between 0 and 1
            )

            glUniform1f(
                glGetUniformLocation(self.shader, l_name + 'strength'),
                light.strength
            )

        # Draw the main object, and recursively draw each of its children
        for obj in self.scene.objects:
            self.draw(obj)

        # Swap the front and back buffers on the window, effectively putting what we just "drew"
        # Onto the screen (whereas previously it only existed in memory)
        # i.e., draw what is in OpenGL framebuffer to window
        pg.display.flip()

        # Do any ticking/timing after flipping
        self.clock.tick(60)  # "Sleep for 1/60 seconds"

    def cleanup(self):
        """Clean up the application"""
        self.scene.cleanup()
        glDeleteProgram(self.shader)

    def quit(self):
        """Quit the application"""
        self.cleanup()
        pg.quit()


def main():
    """The main method"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_width', type=int, default=1280,
                        help='The width of the application window, defaults to 1280')
    parser.add_argument('--window_height', type=int, default=720,
                        help='The height of the application window, defaults to 720')
    parser.add_argument('--fovy', type=float, default=45,
                        help='The field of view (in y direction) in degrees, defaults to 45')
    parser.add_argument('--near', type=float, default=0.1,
                        help='The distance to the near clipping plane, defaults to 0.1')
    parser.add_argument('--far', type=float, default=50,
                        help='The distance to the far clipping plane, defaults to 50')

    parser.add_argument('--vertex_shader_path', default='./shaders/core.vert',
                        help='The path to the vertex shader')
    parser.add_argument('--fragment_shader_path', default='./shaders/core.frag',
                        help='The path to the fragment shader')

    # TODO add option for GPS?
    parser.add_argument('elevation_image', type=str,
                        help='The heightmap to use')
    parser.add_argument('satellite_image', type=str, help='The texture to use')

    # TODO add elevation_zoom level? or always use 16? --quality?
    parser.add_argument('--factor', type=int, default=2,
                        help='The zoom level of the terrain')
    # TODO maybe use desired resolution?

    args = parser.parse_args()

    app = Application(**args.__dict__)
    app.run()


if __name__ == '__main__':
    main()
