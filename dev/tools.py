import os
from glob import glob
import PIL.Image as pilimg
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import kitti_calibration

# from evaluate_depth import STEREO_SCALE_FACTOR
STEREO_SCALE_FACTOR=5.4

def check_files(args):
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        image_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        image_paths = glob(os.path.join(args.image_path, "*.[JjPp]*[PpNn]*[Gg]"))
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    if os.path.isfile(args.calib_path):
        # Only testing on a single image
        calib_paths = [args.calib_path]
    elif os.path.isdir(args.calib_path):
        # Searching folder for images
        calib_paths = glob(os.path.join(args.calib_path, "*.[Tt]*[Xx]*[Tt]"))
    else:
        raise Exception("Can not find args.calib_path: {}".format(args.calib_path))

    if len(calib_paths) == 1:
        calib_paths *= len(image_paths)

    image_paths.sort()
    calib_paths.sort()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    return image_paths, calib_paths, args.output_path

def load_calib(calib_path:str):
    calib = kitti_calibration.Calibration(calib_path)
    return calib


def load_image(image_path:str)->pilimg:
    image = pilimg.open(image_path).convert('RGB')
    return image

def resize_image(image:pilimg, tsize:tuple)->pilimg:
    resized_image = image.resize(tsize, pilimg.LANCZOS)
    return resized_image

def get_colormap(resized_disp_map:np.array)->np.array:
    # resized_disp_map = resized_disp_map.squeeze().cpu().numpy()
    vmax = np.percentile(resized_disp_map, 95)
    normalizer = mpl.colors.Normalize(vmin=resized_disp_map.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(resized_disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormap = pilimg.fromarray(colormapped_im)
    # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
    # name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(file_name))
    # im.save(name_dest_im)
    return colormap

def disp_to_depth(disp:np.array, min_depth:float=0.1, max_depth:float=100.):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_depthamp(disp_resized:np.array):
    scaled_disp, depth = disp_to_depth(disp_resized)
    metric_depth = STEREO_SCALE_FACTOR * depth
    return depth, metric_depth

# image = load_image(image_path="/Users/shinhyeonjun/code/ad-5-final-project-team5/data/monodepth2/000000.png")
# resize_image = resize_image(image=image, tsize=(1024, 320))
# print(resize_image.size)



def gen_rectification(point_cloud:np.array):
    # depth, width, height
    valid_inds = (point_cloud[:, 0] < 120) & \
                 (point_cloud[:, 0] >= 0) & \
                 (point_cloud[:, 1] < 50) & \
                 (point_cloud[:, 1] >= -50) & \
                 (point_cloud[:, 2] < 1.5) & \
                 (point_cloud[:, 2] >= -2.5)
    return point_cloud[valid_inds]



def spartify(point_cloud, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """
    
    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], point_cloud[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map
def project_depth_to_cam_frame_points(depth):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    cam_frame_points = np.stack([c, r, depth])
    cam_frame_points = cam_frame_points.reshape((3, -1))
    cam_frame_points = cam_frame_points.T
    return cam_frame_points

def project_cam_frame_to_lidar_frame(cam_frame_points, calib, max_high):
    cloud = calib.project_image_to_velo(cam_frame_points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]