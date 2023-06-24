import numpy as np
import json
import sys


def main_obj_cut(json_str,obj_file, output_file, r_threshold_multiplier=1.1, z_threshold_multiplier=(0,.9), r_upper_limit = 2,phi_thresholds_offset = (-30,30) ,verbose = False):
    if verbose: print(f'doing main_obj_cut with json_str={json_str}, obj_file={obj_file},output_file={output_file},  r_threshold_multiplier={r_threshold_multiplier}, z_threshold_multiplier={z_threshold_multiplier} ,r_upper_limit={r_upper_limit}, phi_thresholds_offset={phi_thresholds_offset}' )
    cut_reqs = json.loads(json_str)
    phi_thresholds_offset = [angle/180*np.pi for angle in phi_thresholds_offset]
    # if a location should be removed
    def should_cut(vec3, doprint = False):
        transform=np.array(cut_reqs['transform'])
        newp=transform@vec3
        if doprint:print(newp)
        zmin,zmax = cut_reqs['z_range']
        z_new_threshold = (zmin +z_threshold_multiplier[0]*(zmax-zmin),
                        zmin + z_threshold_multiplier[1]*(zmax-zmin))
        if newp[2] < z_new_threshold[0] or newp[2] > z_new_threshold[1] :return True
        newp[:2]-=cut_reqs['circle'][:2]
        dist = np.linalg.norm(newp[:2])
        if doprint:print(dist) 
        if dist<cut_reqs['circle'][2]*r_threshold_multiplier:return False
        if dist>cut_reqs['circle'][2]*r_upper_limit:return True

        phi = np.arctan2(newp[1], newp[0])
        if doprint:print(phi)
        if cut_reqs['phi_range'][1]>np.pi:
            if phi<0:phi+=np.pi*2
        if (phi<cut_reqs['phi_range'][1]+phi_thresholds_offset[1]
                and
                phi>cut_reqs['phi_range'][0] + phi_thresholds_offset[0]):
            return True
        return False
    
    def parse_line_f(f_line):
        toks_l=f_line.split()
        if not toks_l[0]=='f':return []
        v_s =[int(toks.split('/')[0]) for toks in toks_l[1:]]
        return v_s
    # if a line of faces(triangle) match criteria of removing
    def should_remove_f(f_line):
        v_s = parse_line_f(f_line)
        for i in v_s:
            if v_cut_bool[i-1]: return True
        return False
    def should_remove_for_all_mesh(f_line):
        '''here we remove all contents of vertice, uv index and faces '''
        toks_l=f_line.split()
        if toks_l[0] in ['f', 'vt', 'v' ]:
            return True
        return False
    
    # cut_reqs

    # prepare to cut. load obj file first
    with open (obj_file) as f:
        lines = f.readlines()
    obj_toks = [obj.split() for obj in lines]
    keywords = ['v', 'vt', 'f']
    offsets = {}
    v_lst=[]
    vt_lst = []
    f_lst = []
    kw_lsts = {'v':v_lst, 'vt': vt_lst, 'f':f_lst}
    for i,toks in enumerate(obj_toks):
        tp = toks[0]
        if not toks[0] in keywords:continue
        if tp not in offsets:
            offsets[tp]=i
        kw_lsts[tp].append(toks[1:])

    v_lst=np.array(v_lst, dtype = float)
    vt_lst = np.array(vt_lst, dtype = float)
    # apply should_cut to all vertices in v_lst
    v_cut_bool = [should_cut(vec) for vec in v_lst]
    # print ("all_vertices, cutted, remained==",len(v_cut_bool),v_cut_bool.count(True),v_cut_bool.count(False), sep = ', ')
    # print('in generall,we expect cutted>remained')
    remove_ratio = v_cut_bool.count(True)/len(v_cut_bool)
    print (f'all {len(v_cut_bool)} all_vertices. {remove_ratio*100}% cutted. this percentage should be higher than 50%')
    mesh_clear= remove_ratio >0.99
        
    def comment_if_bad(lines):
        for l in lines:
            if should_remove_f(l):
                yield '#'+l
            else:
                yield l
    def remove_if_bad(lines):
        for l in lines:
            if should_remove_f(l):
                pass
            else:
                yield l
    def remove_all_meshes(lines):
        for l in lines:
            if should_remove_for_all_mesh(l):
                pass
            else:
                yield l
    
    with open (output_file, 'w') as f:
        if mesh_clear:
            print("a empty mesh was created")
            f.writelines(remove_all_meshes(lines))
        else:
            f.writelines(remove_if_bad(lines))
#def main(json_str,obj_file, output_file, r_threshold_multiplier=1.1, z_threshold_multiplier=(0,.9) ):
def usage():
    print('Usage: python obj_cut.py cut_range_file obj_to_cut obj_to_save [r_threshold_inner] [z_thresholds] [r_upper_limit] [phi_thresholds_offset]')
    print('\tcut_range_file should be generated with obj_cut.py')
    print('\tobj_to_cut be a .obj file')
    print('\tr_threshold_inner: default 1.1, relative to cam_array')
    print('\tz_thresholds: default pair 0 1, relative to cam_array')
    print('\tr_upper_limit: default  2, relative to cam_array')
    print('\tphi_thresholds_offset: in degree, default pair -30 30, relative to cam_array')

if __name__ == '__main__':
    r_threshold=1.1
    z_thresholds=(0,1)
    r_upper_limit=2
    phi_thresholds_offset=(-30,30)
    try:
        args =sys.argv[1:]
        cut_range_file, obj_to_cut, obj_to_save = args[:3]
        if len(args)>3:
            r_threshold = float(args[3])
        if len(args)>5:
            z_thresholds = float(args[4]), float(args[5])
        if len(args)>6:
            r_upper_limit = float(args[6])
        if len(args)>8:
            phi_thresholds_offset = float(args[7]), float(args[8])
        with open (cut_range_file) as f:
            json_file = f.read()
    except:
        usage()
        exit()
    main_obj_cut(json_file,obj_to_cut,obj_to_save,r_threshold,z_thresholds ,r_upper_limit,phi_thresholds_offset)