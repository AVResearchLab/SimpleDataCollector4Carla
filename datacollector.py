#### Optimization Needed Based on Your Own Needs
##### Note: This script is only for an example and it is not correct for collecting data except for depth image!
## Utilities:
# conda activate carla
# pip install tensorflow==1.14
# pip install protobuf==3.20.*
# pip3 install -r ../../../../PythonAPI/carla/requirements.txt
# export PYTHONPATH=/home/alex/Documents/xxxxxxx/UMCarla-VERSION.9.13/PythonAPI/carla
# cd evaluation
# python3 datacollector.py --sync -m AU -l

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import shutil
import sys
import re
import weakref
from PIL import Image
import queue
import cv2

try:
    import pygame
    from pygame.locals import KMOD_CTRL, K_ESCAPE, K_q
except ImportError:
    raise RuntimeError("Cannot import Pygame, make sure Pygame package is installed")

try:
    import numpy as np
    import numpy.random as random
except ImportError:
    raise RuntimeError("Cannot import numpy, make sure numpy package is installed")

# Find CARLA module
try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass

# Add PythonAPI for release mode
try:
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "carla")
    )
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

LABEL_COLORS = np.array(
    [
        (255, 255, 255),  # None
        (70, 70, 70),  # Building
        (100, 40, 40),  # Fences
        (55, 90, 80),  # Other
        (220, 20, 60),  # Pedestrian
        (153, 153, 153),  # Pole
        (157, 234, 50),  # RoadLines
        (128, 64, 128),  # Road
        (244, 35, 232),  # Sidewalk
        (107, 142, 35),  # Vegetation
        (0, 0, 142),  # Vehicle
        (102, 102, 156),  # Wall
        (220, 220, 0),  # TrafficSign
        (70, 130, 180),  # Sky
        (81, 0, 81),  # Ground
        (150, 100, 100),  # Bridge
        (230, 150, 140),  # RailTrack
        (180, 165, 180),  # GuardRail
        (250, 170, 30),  # TrafficLight
        (110, 190, 160),  # Static
        (170, 120, 50),  # Dynamic
        (45, 60, 150),  # Water
        (145, 170, 100),  # Terrain
    ]
)


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    """Method to find weather presets"""
    def name(x):
        return " ".join(m.group(0) for m in re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================


class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print(
                "  Make sure it exists, has the same name of your town, and is correct."
            )
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self._maps_name = args.map  # Load map mentioned in command line
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 1
        )
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter)
        )
        blueprint.set_attribute("role_name", "hero")
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            spawn_point = spawn_points[0]  # Same start point for every simulation
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(
            self.player, self.hud, self._gamma, self._maps_name
        )
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.sensor_nodes = [None] * 4
        for idx_sensor in range(len(self.camera_manager.sensor_nodes)):
            self.camera_manager.set_sensor1(idx_sensor)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None
        for sensor_node in self.camera_manager.sensor_nodes:
            sensor_node.destroy()

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- Synchronous mode -----------------------------------------------------------
# ==============================================================================


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world.world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get("fps", 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(
            carla.WorldSettings(
                no_rendering_mode=False,
                synchronous_mode=True,
                fixed_delta_seconds=self.delta_seconds,
            )
        )

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height, town_map):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.current_map = town_map

    def on_world_tick(self, timestamp):
        """Gets information from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        self.pose_collector(transform)
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = "N" if abs(transform.rotation.yaw) < 89.5 else ""
        heading += "S" if abs(transform.rotation.yaw) > 90.5 else ""
        heading += "E" if 179.5 > transform.rotation.yaw > 0.5 else ""
        heading += "W" if -0.5 > transform.rotation.yaw > -179.5 else ""
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter("vehicle.audi.a2")
        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name.split("/")[-1],
            "Simulation time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            "Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (transform.rotation.yaw, heading),
            "Location:% 20s"
            % ("(% 5.1f, % 5.1f)" % (transform.location.x, transform.location.y)),
            "GNSS:% 24s"
            % ("(% 2.6f, % 3.6f)" % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            "Height:  % 18.0f m" % transform.location.z,
            "",
        ]
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", control.throttle, 0.0, 1.0),
                ("Steer:", control.steer, -1.0, 1.0),
                ("Brake:", control.brake, 0.0, 1.0),
                ("Reverse:", control.reverse),
                ("Hand brake:", control.hand_brake),
                ("Manual:", control.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(control.gear, control.gear),
            ]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ("Speed:", control.speed, 0.0, 5.556),
                ("Jump:", control.jump),
            ]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
        def dist(l):
            return math.sqrt(
                (l.x - transform.location.x) ** 2
                + (l.y - transform.location.y) ** 2
                + (l.z - transform.location.z) ** 2
            )
        vehicles = [
            (dist(x.get_location()), x) for x in vehicles if x.id != world.player.id
        ]
        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append("% 4dm %s" % (dist, vehicle_type))

    def pose_collector(self, transform):
        frame_id = self.frame
        pose_matrix_4 = transform.get_matrix()
        pose_matrix_3 = pose_matrix_4[0:3][:]
        pose_matrix_1 = [j for sub in pose_matrix_3 for j in sub]
        pose_matrix_1.append(frame_id)
        filepath = "./dataset/{town}/poses.txt".format(town=self.current_map)
        textfile = open(filepath, "a")
        for element in pose_matrix_1:
            element = float("{0:.6g}".format(element))
            element_str = str(element)
            textfile.write(element_str + " ")
        textfile.write("\n")
        textfile.close

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (fig * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================

class FadingText(object):
    """Class for fading text"""

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(int(500.0 * self.seconds_left))

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)



# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

class CollisionSensor(object):
    """Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        self.sensor.listen(lambda event: self._on_collision(event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    def _on_collision(self, event):
        """On collision method"""
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f"Collision with {actor_type}")
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        self.sensor.listen(lambda event: self._on_invasion(event))

    def _on_invasion(self, event):
        """On invasion method"""
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    """Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.spawn_actor(
            blueprint,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=self._parent,
        )
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- 3D - 2D projection -------------------------------------------------------------
# ==============================================================================


def load_point_cloud_data(args, snapshot):
    """Load point cloud data from a PLY file."""
    filename = f"./dataset/{args.map}/lidar_semseg/raw_data/ply_files/{snapshot}_3.ply"
    with open(filename, "rt") as myfile:
        point_cloud_data = [line.split() for line in myfile]
    return np.array(point_cloud_data[10:-1], dtype=np.float32)

def calculate_intensity_loss(depth):
    """Calculate intensity loss based on depth."""
    a = 0.004  # Atmosphere attenuation rate
    return np.exp(-a * depth)

def project_point_cloud(point_cloud_xyz, proj_fov_up, proj_fov_down, proj_W, proj_H):
    """Project point cloud onto a spherical surface."""
    fov_up = proj_fov_up / 180.0 * np.pi
    fov_down = proj_fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)
    depth = np.linalg.norm(point_cloud_xyz[:, :3], axis=1)
    yaw = np.arctan2(point_cloud_xyz[:, 1], point_cloud_xyz[:, 0])
    pitch = np.arcsin(point_cloud_xyz[:, 2] / depth)
    proj_x = np.floor(0.5 * ((yaw / np.pi) + 1.0) * proj_W).astype(np.uint32)
    proj_y = np.floor((1.0 - (pitch + abs(fov_down)) / fov) * proj_H).astype(np.uint32)
    proj_x = np.clip(proj_x, 0, proj_W - 1)
    proj_y = np.clip(proj_y, 0, proj_H - 1)
    return depth, proj_x, proj_y

def create_image_data(proj_sem_color, image_data):
    """Create and save image data."""
    image = Image.fromarray(proj_sem_color.astype(np.uint8))
    image.save(f"./dataset/AU/lidar_semseg/images/{image_data.frame:08d}.png")

def do_range_projection(args, proj_fov_up, proj_fov_down, snapshot, image_data, LABEL_COLORS):
    """Project a point cloud into a spherical projection image."""
    point_cloud_data = load_point_cloud_data(args, snapshot)
    depth, proj_x, proj_y = project_point_cloud(point_cloud_data, proj_fov_up, proj_fov_down, 1024, 64)
    intensity_loss = calculate_intensity_loss(depth)

    proj_H, proj_W = 64, 1024
    proj_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
    proj_cosine = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_sem_label = np.full((proj_H, proj_W), 0, dtype=np.uint32)
    proj_inst_label = np.full((proj_H, proj_W), 0, dtype=np.uint32)
    proj_sem_color = np.full((proj_H, proj_W, 3), 0, dtype=np.float32)
    proj_intensity = np.full((proj_H, proj_W), -1, dtype=np.float32)

    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = point_cloud_data[:, :3]
    proj_cosine[proj_y, proj_x] = point_cloud_data[:, 3]
    proj_sem_label[proj_y, proj_x] = point_cloud_data[:, 5].astype(np.uint32)
    proj_sem_color[proj_y, proj_x] = LABEL_COLORS[proj_sem_label[proj_y, proj_x]]
    proj_inst_label[proj_y, proj_x] = point_cloud_data[:, 4].astype(np.uint32)
    proj_intensity[proj_y, proj_x] = intensity_loss

    create_image_data(proj_sem_color, image_data)

# ==============================================================================
# -- Create input tensor and store in a binary file ----------------------------
# ==============================================================================

def create_input_tensor(args, snapshot, proj_xyz, proj_intensity, proj_range):
    """Create input tensor and save it as a binary file."""
    dir_path = f"./dataset/{args.map}/lidar_semseg/raw_data/binary_files/"
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"{snapshot}_3.bin"
    file_path = os.path.join(dir_path, file_name)

    proj_intensity = np.expand_dims(proj_intensity, axis=2)
    proj_range = np.expand_dims(proj_range, axis=2)
    input_tensor = np.concatenate((proj_xyz, proj_intensity, proj_range), axis=2)
    
    with open(file_path, "wb") as binary_file:
        input_tensor.tofile(binary_file)


# ==============================================================================
# -- Create ground truth and store in a binary file ----------------------------
# ==============================================================================

def create_ground_truth(args, snapshot, proj_sem_label):
    """Create ground truth label file and save it."""
    dir_path = f"./dataset/{args.map}/lidar_semseg/raw_data/ground_truth/"
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"{snapshot}_3.label"
    file_path = os.path.join(dir_path, file_name)

    output_tensor = proj_sem_label & 0xFFFF
    
    with open(file_path, "wb") as label_file:
        output_tensor.tofile(label_file)
    
    updated_ground_truth(args, snapshot, file_path)


# ==============================================================================
# ---Labelling only car and padestrian; Treating other classes as background----
# ==============================================================================

def updated_ground_truth(args, snapshot, filepath_first):
    """Update ground truth label file and save it."""
    dir_path = f"./dataset/{args.map}/lidar_semseg/raw_data/updated_ground_truth/"
    os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"{snapshot}_3_new.label"
    file_path_second = os.path.join(dir_path, file_name)

    original_label = np.fromfile(filepath_first, dtype=np.uint32)
    modified_label = np.where(original_label == 4, 2, np.where(original_label == 10, 1, 0))

    with open(file_path_second, "wb") as label_file:
        mod_output_tensor = modified_label & 0xFFFF
        mod_output_tensor.tofile(label_file)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    """Class for camera management"""

    sensor_node_list = []

    def __init__(self, parent_actor, hud, gamma_correction, map_name):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.town_map = map_name
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=0.0, z=2.1)), attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid),
        ]
        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.semantic_segmentation", cc.Raw, "Camera Semantic Segmentation (Raw)"],
            ["sensor.lidar.ray_cast_semantic", None, "Semantic LIDAR"],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                blp.set_attribute("image_size_x", str(hud.dim[0]))
                blp.set_attribute("image_size_y", str(hud.dim[1]))
                if blp.has_attribute("gamma"):
                    blp.set_attribute("gamma", str(gamma_correction))
            elif item[0].startswith("sensor.lidar.ray_cast_semantic"):
                blp.set_attribute("role_name", "semantic_lidar")
                blp.set_attribute("rotation_frequency", "20.0")
                blp.set_attribute("range", "100.0")
                blp.set_attribute("channels", "64")
                blp.set_attribute("upper_fov", "3.0")
                blp.set_attribute("lower_fov", "-25.0")
                blp.set_attribute("points_per_second", "2600000")  # 660000 1300000
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def set_sensor1(self, idx_sensor, notify=True, force_respawn=False):
        """Set a sensor"""
        needs_respawn = True
        if needs_respawn:
            if self.sensor_nodes[idx_sensor] is not None:
                self.sensor_nodes[idx_sensor].destroy()
            self.sensor_nodes[idx_sensor] = self._parent.get_world().spawn_actor(
                self.sensors[idx_sensor][-1],
                self._camera_transforms[1][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[1][1],
            )
            self.sensor_node_list.append(self.sensor_nodes[idx_sensor])

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.rgb"):
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)



def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array


def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def labels_to_array(image):
    """
    Convert an image containing CARLA semantic segmentation labels to a 2D array
    containing the label of each pixel.
    """
    return to_bgra_array(image)[:, :, 2]


def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],        # None
        1: [70, 70, 70],     # Buildings
        2: [190, 153, 153],  # Fences
        3: [72, 0, 90],      # Other
        4: [220, 20, 60],    # Pedestrians
        5: [153, 153, 153],  # Poles
        6: [157, 234, 50],   # RoadLines
        7: [128, 64, 128],   # Roads
        8: [244, 35, 232],   # Sidewalks
        9: [107, 142, 35],   # Vegetation
        10: [0, 0, 255],     # Vehicles
        11: [102, 102, 156], # Walls
        12: [220, 220, 0]    # TrafficSigns
    }
    array = labels_to_array(image)
    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result


def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    grayscale = np.dot(array[:, :, :3], [256.0 * 256.0, 256.0, 1.0])
    grayscale /= (256.0 * 256.0 * 256.0 - 1.0)
    return grayscale


def depth_to_logarithmic_grayscale(image):
    """
    Convert an image containing CARLA encoded depth-map to a logarithmic
    grayscale image array.
    """
    grayscale = depth_to_array(image)
    # Convert to logarithmic depth.
    logdepth = np.ones(grayscale.shape) + (np.log(grayscale) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    # Expand to three colors.
    return np.repeat(logdepth[:, :, np.newaxis], 3, axis=2)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.load_world(args.map)

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        hud = HUD(args.width, args.height, args.map)
        world = World(client.load_world(args.map), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)

        # Get the sensor list, waypoint, vehicle
        sensor_list = CameraManager.sensor_node_list
        m = world.world.get_map()
        start_pose = spawn_points[0]
        waypoint = m.get_waypoint(start_pose.location)
        clock = pygame.time.Clock()

        # Make directories for storing raw data
        parent_dir = os.getcwd()
        directory_rgb = f"./dataset/{args.map}/camera_rgb/raw_data/binary"
        directory_depth = f"./dataset/{args.map}/camera_depth/raw_data/binary"
        directory_semseg = f"./dataset/{args.map}/camera_semseg/raw_data/binary"
        directory_pcimage = f"./dataset/{args.map}/lidar_semseg/images"
        path_rgb = os.path.join(parent_dir, directory_rgb)
        path_depth = os.path.join(parent_dir, directory_depth)
        path_semseg = os.path.join(parent_dir, directory_semseg)
        path_pcimage = os.path.join(parent_dir, directory_pcimage)
        directories = [path_rgb, path_depth, path_semseg, path_pcimage]
        for directory in directories:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

        SAVE_INTERVAL = 3.14  # Save data once per second
        last_save_time = 0.0
        while True:

            # Create a synchronous mode context.
            with CarlaSyncMode(
                world,
                sensor_list[0],
                sensor_list[1],
                sensor_list[2],
                sensor_list[3],
                fps=10,
            ) as sync_mode:
                while True:
                    if args.sync:
                        clock.tick()
                        elapsed_time = pygame.time.get_ticks()
                        if elapsed_time - last_save_time >= SAVE_INTERVAL:
                            last_save_time = elapsed_time

                            # Advance the simulation and wait for the data.
                            snapshot, image_rgb, image_depth, image_semseg, data_lidar = (
                                sync_mode.tick(timeout=2.0)
                            )

                            # RGB
                            image_rgb.convert(cc.Raw)
                            array_rgb = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
                            array_rgb = np.reshape(array_rgb, (image_rgb.height, image_rgb.width, 4))
                            array_rgb = array_rgb[:, :, :3]
                            array_rgb = array_rgb[:, :, ::-1]
                            filename_rgb = "{frame_id}.npy".format(frame_id=snapshot)
                            np.save(os.path.join(path_rgb, filename_rgb), array_rgb)
                            image_rgb.save_to_disk("./dataset/{town}/camera_rgb/images/{frame_id}_{sensor_id}".format(town=args.map, frame_id=snapshot, sensor_id=0))
                            
                            # Depth
                            image_depth.convert(cc.Raw)
                            array_depth = depth_to_array(image_depth)
                            depth_data_normalized = array_depth / 0.01 # Convert depth image to centimeters
                            # Apply colormap for visualization
                            colormap = cv2.COLORMAP_JET # blue: low; green: intermediate; red: high
                            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_data_normalized, alpha=255.0), colormap)
                            filename_depth = f"{snapshot}.png"
                            # add a line of code here to check if the folder is not created, then create the folders
                            directory_path = os.path.join("./dataset/", args.map, "camera_depth", "images")
                            if not os.path.exists(directory_path):
                                os.makedirs(directory_path)
                            filename_depth = f"{snapshot}.png"
                            depth_image_filepath = os.path.join(directory_path, filename_depth)
                            cv2.imwrite(depth_image_filepath, depth_colormap)
                            
                            # SemanticSegmentation
                            image_semseg.convert(cc.Raw)
                            array_semseg = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8"))
                            array_semseg = np.reshape(array_semseg, (image_semseg.height, image_semseg.width, 4))
                            array_semseg = array_semseg[:, :, :3]
                            array_semseg = array_semseg[:, :, 2:3]
                            filename_semseg = "{frame_id}.npy".format(frame_id=snapshot)
                            np.save(os.path.join(path_semseg, filename_semseg), array_semseg)
                            image_semseg.save_to_disk("./dataset/{town}/camera_semseg/images/{frame_id}_{sensor_id}".format(town=args.map, frame_id=snapshot, sensor_id=2), cc.CityScapesPalette,)

                            data_lidar.save_to_disk("./dataset/{town}/lidar_semseg/raw_data/ply_files/{frame_id}_{sensor_id}".format(town=args.map, frame_id=snapshot, sensor_id=3))
                            do_range_projection(args, 3, -25, snapshot, image_rgb, LABEL_COLORS)

                            # Choose the next waypoint and update the car location.
                            waypoint = random.choice(waypoint.next(1.5))
                    else:
                        world.world.wait_for_tick()
                    if controller.parse_events():
                        return

                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()

                    if agent.done():
                        if args.loop:
                            agent.set_destination(random.choice(spawn_points).location)
                            world.hud.notification(
                                "The target has been reached, searching for another target",
                                seconds=4.0,
                            )
                            print("The target has been reached, searching for another target")
                        else:
                            print("The target has been reached, stopping the simulation")
                            break

                    control = agent.run_step()
                    control.manual_gear_shift = False
                    world.player.apply_control(control)

    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

def save_data(snapshot, image_rgb, image_depth, image_semseg, data_lidar, args):
    """Save sensor data to disk."""
    save_image_rgb(snapshot, image_rgb, args)
    save_image_depth(snapshot, image_depth, args)
    save_image_semseg(snapshot, image_semseg, args)
    save_data_lidar(snapshot, data_lidar, args)

def save_image_rgb(snapshot, image_rgb, args):
    """Save RGB image to disk."""
    image_rgb.convert(cc.Raw)
    array_rgb = np.frombuffer(image_rgb.raw_data, dtype=np.dtype("uint8"))
    array_rgb = np.reshape(array_rgb, (image_rgb.height, image_rgb.width, 4))
    array_rgb = array_rgb[:, :, :3]
    array_rgb = array_rgb[:, :, ::-1]
    filename_rgb = f"{snapshot}.npy"
    directory_path = os.path.join(args.map, "camera_rgb", "raw_data", "binary")
    os.makedirs(directory_path, exist_ok=True)
    np.save(os.path.join(directory_path, filename_rgb), array_rgb)

def save_image_depth(snapshot, image_depth, args):
    """Save depth image to disk."""
    image_depth.convert(cc.Raw)
    array_depth = np.frombuffer(image_depth.raw_data, dtype=np.dtype("uint8"))
    array_depth = np.reshape(array_depth, (image_depth.height, image_depth.width, 4))
    array_depth = array_depth.astype(np.float32)
    array_depth = array_depth[:, :, :3]
    array_depth = array_depth[:, :, ::-1]
    array_depth = (
        array_depth[:, :, 0:1]
        + 256 * array_depth[:, :, 1:2]
        + 256 * 256 * array_depth[:, :, 2:3]
    )
    array_depth /= 16777215
    array_depth *= 1000 * 100
    filename_depth = f"{snapshot}.npy"
    directory_path = os.path.join(args.map, "camera_rgb", "raw_data", "binary")
    os.makedirs(directory_path, exist_ok=True)
    np.save(os.path.join(directory_path, filename_depth), array_depth)

def save_image_semseg(snapshot, image_semseg, args):
    """Save semantic segmentation image to disk."""
    image_semseg.convert(cc.Raw)
    array_semseg = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8"))
    array_semseg = np.reshape(array_semseg, (image_semseg.height, image_semseg.width, 4))
    array_semseg = array_semseg[:, :, :3]
    array_semseg = array_semseg[:, :, 2:3]
    filename_semseg = f"{snapshot}.npy"
    directory_path = os.path.join(args.map, "camera_rgb", "raw_data", "binary")
    os.makedirs(directory_path, exist_ok=True)
    np.save(os.path.join(directory_path, filename_semseg), array_semseg)

def save_data_lidar(snapshot, data_lidar, args):
    """Save lidar data to disk."""
    data_lidar.save_to_disk(
        os.path.join(args.map, "lidar_semseg", "raw_data", "ply_files", f"{snapshot}_3")
    )


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

import argparse
import logging

def main():
    """Main method"""

    # Argument parsing
    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="Print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="Window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--sync",
        action="store_true",
        help="Enable synchronous mode execution",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.audi.a2",
        help='Actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "--gamma",
        default=2.2,
        type=float,
        help="Gamma correction of the camera (default: 2.2)",
    )
    argparser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        dest="loop",
        help="Sets a new random destination upon reaching the previous one (default: False)",
    )
    argparser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Behavior", "Basic"],
        help="Select which agent to run (default: Behavior)",
        default="Behavior",
    )
    argparser.add_argument(
        "-b",
        "--behavior",
        type=str,
        choices=["cautious", "normal", "aggressive"],
        help="Choose one of the possible agent behaviors (default: normal)",
        default="normal",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        help="Set seed for repeating executions (default: None)",
        default=None,
        type=int,
    )
    argparser.add_argument(
        "-m",
        "--map",
        required=True,
        help="Load a new map",
    )
    argparser.add_argument(
        "-d",
        "--dot-extent",
        metavar="SIZE",
        default=2,
        type=int,
        help="Visualization dot extent in pixels (Recommended: 1-4) (default: 2)",
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]
    args.dot_extent -= 1

    # Set logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("Listening to server %s:%s", args.host, args.port)

    print(__doc__)

    try:
        # Start the main loop
        game_loop(args)

    except KeyboardInterrupt:
        logging.info("Cancelled by user. Exiting...")

if __name__ == "__main__":
    pygame.quit()
    main()
