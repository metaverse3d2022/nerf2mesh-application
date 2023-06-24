import mesh_cut.cam_fit as cf
import mesh_cut.obj_cut as oc

import sys
from nerf.opt import *
import os

if __name__ == '__main__':
    verbose=False
    r_threshold=0.9
    z_thresholds=(-1.1, 1.65)
    r_upper_limit=1.3
    phi_thresholds_offset=(-25,25)

    # generate poses
    opt = get_opt()
    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # 'nerf
        from nerf.provider import NeRFDataset
    device = 'cpu'
    dataset = NeRFDataset(opt, device=device, type='train')
    poses = dataset.poses.tolist()

    for fff in os.listdir(opt.obj_cut_dir):
        if fff.strip().split(".")[-1] == "obj":
            old_obj_file = os.path.join(opt.obj_cut_dir, fff)
            json_str = cf.main(cam_pose = poses, verbose = False,  special_mode = True)
            oc.main_obj_cut(
                    json_str, old_obj_file, old_obj_file, 
                    r_threshold, z_thresholds, r_upper_limit, 
                    phi_thresholds_offset, verbose = verbose)
