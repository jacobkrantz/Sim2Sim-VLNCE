"""dynamic configuration modifiers"""

from copy import deepcopy

import numpy as np
from habitat import Config as CN


def add_pano_sensors_to_config(config: CN) -> CN:
    """Dynamically adds RGB and Depth cameras to config.TASK_CONFIG, forming
    an N-frame panorama. The PanoRGB and PanoDepth observation transformers
    can be used to stack these frames together.
    """
    num_cameras = config.TASK_CONFIG.TASK.PANO_ROTATIONS

    config.defrost()
    orient = [(0, np.pi * 2 / num_cameras * i, 0) for i in range(num_cameras)]
    sensor_uuids = ["rgb"]
    if "RGB_SENSOR" in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, num_cameras):
            camera_template = f"RGB_{camera_id}"
            camera_config = deepcopy(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]

            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)
            setattr(
                config.TASK_CONFIG.SIMULATOR, camera_template, camera_config
            )
            config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(
                camera_template
            )

    sensor_uuids = ["depth"]
    if "DEPTH_SENSOR" in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS:
        config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.ORIENTATION = orient[0]
        for camera_id in range(1, num_cameras):
            camera_template = f"DEPTH_{camera_id}"
            camera_config = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
            camera_config.ORIENTATION = orient[camera_id]
            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)

            setattr(
                config.TASK_CONFIG.SIMULATOR, camera_template, camera_config
            )
            config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(
                camera_template
            )

    config.SENSORS = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    config.freeze()
    return config


def add_vln_pano_sensors_to_config(config: CN) -> CN:
    """Dynamically adds RGB cameras to config.TASK_CONFIG, forming a panorama
    observation space of 12 headings and 3 elevations that matches the
    36-frame VLN panorama.
    """
    assert "RGB_SENSOR" in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS

    HEADINGS = [0.0] + [np.pi * 2 / 12 * i for i in range(1, 12)][::-1]
    ELEVATIONS = [np.deg2rad(e) for e in [-30, 0, 30]]

    config.defrost()
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.remove("RGB_SENSOR")

    sensor_uuids = []
    camera_idx = 0
    for elevation in ELEVATIONS:
        for heading in HEADINGS:
            uuid_template = f"RGB_{camera_idx}"
            camera_cfg = deepcopy(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
            camera_cfg.UUID = uuid_template.lower()
            sensor_uuids.append(uuid_template.lower())

            camera_cfg.ORIENTATION = [elevation, heading, 0.0]
            setattr(config.TASK_CONFIG.SIMULATOR, uuid_template, camera_cfg)
            config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(uuid_template)
            camera_idx += 1

    config.SENSORS = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    config.freeze()
    return config


def add_2d_laser_to_config(config: CN) -> CN:
    """Constructs 4 depth patches that together mimic a 360 degree 2D laser
    scan. Each camera has an HFOV of 90.
    """
    sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    if not config.TASK_CONFIG.TASK.LASER_2D.ENABLED:
        return config

    C = 4
    HEADINGS = [0.0] + [np.pi * 2 / C * i for i in range(1, C)][::-1]

    config.defrost()
    laser_cfg = config.TASK_CONFIG.TASK.LASER_2D

    camera_cfg = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
    camera_cfg.WIDTH = laser_cfg.WIDTH
    camera_cfg.HEIGHT = laser_cfg.HEIGHT
    camera_cfg.POSITION = [0, laser_cfg.SENSOR_HEIGHT, 0]
    camera_cfg.HFOV = 90
    camera_cfg.MAX_DEPTH = laser_cfg.MAX_DEPTH
    camera_cfg.NORMALIZE_DEPTH = laser_cfg.NORMALIZE_DEPTH

    for camera_id in range(C):
        template = f"LASER_2D_{camera_id}"
        cfg_i = deepcopy(camera_cfg)
        cfg_i.ORIENTATION = [0.0, HEADINGS[camera_id], 0.0]
        cfg_i.UUID = template.lower()
        setattr(config.TASK_CONFIG.SIMULATOR, template, cfg_i)
        sensors.append(template)

    config.SENSORS = sensors
    config.freeze()
    return config


def add_depth_for_map(config: CN) -> CN:
    """dynamically adds a depth camera for mapping purposes that has the same
    config as https://github.com/meera1hahn/NRNS.
    """
    if not config.LOCAL_POLICY:
        return config

    config.defrost()

    template = "MAPPING_DEPTH"
    camera_cfg = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
    camera_cfg.WIDTH = 640
    camera_cfg.HEIGHT = 480
    camera_cfg.POSITION = [0, 1.25, 0]
    camera_cfg.ORIENTATION = [0.0, 0.0, 0.0]
    camera_cfg.HFOV = 120
    camera_cfg.MAX_DEPTH = 10.0
    camera_cfg.MIN_DEPTH = 0.0
    camera_cfg.NORMALIZE_DEPTH = True
    camera_cfg.UUID = template.lower()
    setattr(config.TASK_CONFIG.SIMULATOR, template, camera_cfg)

    sensors = config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
    sensors.append(template)

    config.SENSORS = sensors
    config.freeze()
    return config
