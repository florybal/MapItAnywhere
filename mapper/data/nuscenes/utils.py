import os
import numpy as np
from shapely import geometry, affinity
from pyquaternion import Quaternion
import cv2

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.utils.data_classes import LidarPointCloud

from nuscenes.map_expansion.map_api import NuScenesMap
from shapely.strtree import STRtree
from collections import OrderedDict
import torch

def decode_binary_labels(labels, nclass):
    bits = torch.pow(2, torch.arange(nclass))
    return (labels & bits.view(-1, 1, 1)) > 0

def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)

def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors

CAMERA_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']

NUSCENES_CLASS_NAMES = [
    'drivable_area', 'ped_crossing', 'walkway', 'carpark', 'car', 'truck', 
    'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
    'bicycle', 'traffic_cone', 'barrier'
]

STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']

def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)

        
        # Store as an R-Tree for fast intersection queries
        tree = STRtree(polygons)
        # Some Shapely versions return integer indices from STRtree.query().
        # Attach the original polygon list to the tree so callers can recover
        # the geometry when an index is returned.
        try:
            tree._input_polygons = polygons
        except Exception:
            # best-effort: if we can't attach, continue — callers will try other fallbacks
            pass
        map_data[layer] = tree
    
    return map_data

def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != '':
        sample = nuscenes.get('sample', sample_token)
        yield sample
        sample_token = sample['next']
    

def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):

    # Render each layer sequentially
    layers = [get_layer_mask(nuscenes, polys, sample_data, extents, 
              resolution) for layer, polys in map_data.items()]

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)),
                    dtype=np.uint8)

    # Find all polygons which intersect with the area of interest. STRtree.query
    # may return geometry objects or integer indices depending on Shapely version.
    for hit in polygons.query(map_patch):
        polygon = hit
        # Resolve integer indices to actual geometries when needed
        if isinstance(polygon, (int, np.integer)):
            if hasattr(polygons, '_input_polygons'):
                polygon = polygons._input_polygons[int(polygon)]
            elif hasattr(polygons, 'geometries'):
                # Some STRtree implementations expose geometries
                polygon = polygons.geometries[int(polygon)]
            else:
                # Can't resolve the index -> skip
                continue

        if polygon is None:
            continue

        # Intersect with the query patch and skip empty results
        polygon = polygon.intersection(map_patch)
        if polygon.is_empty:
            continue

        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)
    
    return mask




def get_object_masks(nuscenes, sample_data, extents, resolution):

    # Initialize object masks
    nclass = len(DETECTION_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data['token']):

        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in DETECTION_NAMES:
            class_id = -1
        else:
            class_id = DETECTION_NAMES.index(det_name)
        
        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)
    
    # Return boolean mask (avoid deprecated np.bool alias)
    return masks.astype(bool)


def get_sensor_transform(nuscenes, sample_data):

    # Load sensor transform data
    sensor = nuscenes.get(
        'calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get('ego_pose', sample_data['ego_pose_token'])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def load_point_cloud(nuscenes, sample_data):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record['rotation']).rotation_matrix
    transform[:3, 3] = np.array(record['translation'])
    return transform


def render_shapely_polygon(mask, polygon, extents, resolution):

    # Defensive: skip empty geometries
    if polygon is None:
        return
    if getattr(polygon, 'is_empty', False):
        return

    # If this is a simple polygon, render normally
    if getattr(polygon, 'geom_type', None) == 'Polygon':

        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
        return

    # Handle compound shapes (MultiPolygon, GeometryCollection)
    try:
        for poly in polygon.geoms:
            render_shapely_polygon(mask, poly, extents, resolution)
    except Exception:
        # Fallback: try iteration (some objects are iterable)
        try:
            for poly in polygon:
                render_shapely_polygon(mask, poly, extents, resolution)
        except Exception:
            # Unknown geometry type — skip
            return