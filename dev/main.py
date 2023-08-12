import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
import numpy as np

import tools
from monodepth import Monodepth

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--calib_path', type=str,
                        help='path to a test calib_path or folder of calib_path', required=True)
    parser.add_argument('--output_path', type=str,
                        help='path to a output folder', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument("--cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # print(args)
    image_paths, calib_paths, output_path = tools.check_files(args)

    monodepth = Monodepth(args=args)
    for idx, (image_path, calib_path) in enumerate(zip(image_paths, calib_paths)):
        output_file_name = os.path.basename(image_path)[:6]

        calib = tools.load_calib(calib_path=calib_path)
        disp_map, resized_disp_map = monodepth.get_disparity_map(image_path=image_path)
        depthmap, metric_depthmap = tools.get_depthamp(resized_disp_map)

        cam_frame_points = tools.project_depth_to_cam_frame_points(metric_depthmap)
        lidar_frame_points = tools.project_cam_frame_to_lidar_frame(cam_frame_points, calib, 1)

        lidar = np.concatenate( (lidar_frame_points, np.ones((len(lidar_frame_points), 1)) ), 1)
        lidar = lidar.astype(np.float32)
        lidar_rectified = tools.gen_rectification(lidar).astype(np.float32)
        # print(lidar_rectified)
        lidar_spartified = tools.spartify(lidar_rectified).astype(np.float32)

        colormap_image = tools.get_colormap(resized_disp_map)
        name_dest_im = os.path.join(output_path, "{}_disp.jpeg".format(output_file_name))
        colormap_image.save(name_dest_im)
        
        name_dest_npy = os.path.join(output_path, "{}_disp.npy".format(output_file_name))
        np.save(name_dest_npy, disp_map)
        name_resized_dest_npy = os.path.join(output_path, "{}_resized_disp.npy".format(output_file_name))
        np.save(name_resized_dest_npy, resized_disp_map)
        
        save_file_name = '{}_pllidar.bin'.format(os.path.join(args.output_path, output_file_name))
        lidar.tofile(save_file_name)
        save_file_name = '{}_pllidar_rectified.bin'.format(os.path.join(args.output_path, output_file_name))
        lidar_rectified.tofile(save_file_name)
        save_file_name = '{}_pllidar_spartified.bin'.format(os.path.join(args.output_path, output_file_name))
        lidar_spartified.tofile(save_file_name)

        print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(image_paths)))
        print("   - {} file compelete".format(output_file_name))

    print('-> Done!')