#


import numpy as np
from sklearn.cluster import KMeans

from circle_fit import taubinSVD
import json

def fit_plane(points):
    """
    Fit a plane to a set of 3D points using least squares.

    Args:
        points: A numpy array of shape (n, 3) representing n points in 3D space.

    Returns:
        A tuple (p, n) representing the fitted plane, where p is a point on
        the plane and n is the normal vector of the plane.
    """

    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Subtract the centroid from the points to obtain centered points
    centered_points = points - centroid

    # Compute the singular value decomposition of the centered points
    _, _, v = np.linalg.svd(centered_points)

    # The last column of v is the normal vector of the plane
    normal = v[-1]

    # The point on the plane is the centroid
    point = centroid

    return point, normal

def fit_planes(points, num_planes):
    """
    Fit multiple planes to a set of 3D points using K-Means clustering and least squares.

    Args:
        points: A numpy array of shape (n, 3) representing n points in 3D space.
        num_planes: An integer representing the number of planes to fit.

    Returns:
        A list of tuples, each representing a fitted plane, where each tuple contains
        a point on the plane and the normal vector of the plane.
    """

    # Perform K-Means clustering to group the points into num_planes clusters
    kmeans = KMeans(n_clusters=num_planes, random_state=0).fit(points)

    # Initialize an empty list to store the planes
    planes = []

    # Fit a plane to each cluster of points
    for i in range(num_planes):
        cluster_points = points[kmeans.labels_ == i]
        plane = fit_plane(cluster_points)
        planes.append(plane)

    return planes


def make_z_axis(vector):
    # # Usage:
    # transform = make_z_axis([1, 2, 3])
    # transform
    # Normalize the z-axis vector
    z_axis = vector / np.linalg.norm(vector)
    
    # Project x and y axes onto plane orthogonal to z-axis
    x_axis = project_on_plane(z_axis, [1, 0, 0]) 
    y_axis = project_on_plane(z_axis, [0, 1, 0])
    
    # Ensure x and y axes are orthogonal
    if not are_orthogonal(x_axis, y_axis):
        y_axis = np.cross(z_axis, x_axis) / np.linalg.norm(np.cross(z_axis, x_axis))
        
    # Create transformation matrix 
    transform = np.array([
        [x_axis[0], x_axis[1], x_axis[2], 0],
        [y_axis[0], y_axis[1], y_axis[2], 0],
        [z_axis[0], z_axis[1], z_axis[2], 0],
        [0, 0, 0, 1]
    ])
    
    return np.array(transform[:3,:3])

def project_on_plane(normal, vector):
    # Project vector onto plane orthogonal to normal vector
    return vector - normal * np.dot(normal, vector) / np.dot(normal, normal)  

def are_orthogonal(a, b):
    # Check if two vectors are orthogonal
    return np.abs(np.dot(a, b)) < 1e-6

def find_best_range_phi(phis):
    # phis = [-2,-1,1,2,3.]
    phis=sorted(phis)
    phis.append(phis[0])
    phis=np.array(phis)
    segs=phis[1:]-phis[:-1]

    segs=np.where(segs>0,segs, segs+2*np.pi)# segs now proper
    largest=segs.argmax()
    # print(segs,largest)
    if not largest == len(segs)-1:
        phis[0:largest+1]+=+2*np.pi
        # print(sorted(phis[:-1]))
    return phis[:-1].min(),phis[:-1].max()

def find_range_z(zs):
    return zs.min(), zs.max()



def usage():
    print('Usage: python cam_fit.py [--verbose] file_of_cam.json [cut_range_output_file]')
    print('note it requires file_of_cam a list of mat44')
    print('will write a file cam_array_explain.png if verbose')

def main(cam_pose, verbose = False,cut_range_output_file = None ,special_mode=False):
    # load files and get poses
    # with open (cam_file) as f:
    #    transformsjson2=json.load(f)
    rot_and_poses = cam_pose
    pure_poses=np.array([np.array(p)[0:3,3] for p in rot_and_poses])

    # fit 3 cam planes
    planes_list = fit_planes(pure_poses, num_planes=3)
    if (verbose):print(planes_list)

    # use 2 planes to find a z-axis
    planes_list=np.array(planes_list)
    z0 = np.cross(planes_list[0][1],planes_list[1][1])
    z1 = np.cross(planes_list[0][1],planes_list[2][1])

    z0/=np.linalg.norm(z0)
    z1/=np.linalg.norm(z1)

    if np.dot(z0, (0,0,1))<0: z0 *=-1
    if np.dot(z1, (0,0,1))<0: z1 *=-1

    # a transform mat for getting real z
    transform = make_z_axis(z1)
    if (verbose):print(transform)

    
    transformed_poses = (transform@pure_poses.T).T

    # circle fit
    points = transformed_poses[:,:2]  # N x 2 array of (x, y) points
    xc, yc, r, sigma = taubinSVD(points)

    # after all fits, yield in new axis z range and phi range for cam_array 
    dirs=points - [xc,yc]
    # fig = plt.figure()
    # ax = fig.add_subplot()
    phis=np.arctan2(dirs[:,1], dirs[:,0]) # arctan2 range be -pi pi
    begin, end =find_best_range_phi(phis)
    zrange = find_range_z(transformed_poses[:,2])
    # draw to a picture 
    if verbose:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(transformed_poses[:,0], transformed_poses[:,1])
        # Create the circle patch 
        def draw_circle(xc,yc,r):
            circle = Circle((xc,yc), r,fill=False )
            ax.add_patch(circle)
        draw_circle(xc,yc,r)
        draw_circle(xc,yc,0.9*r)
        center = np.array([xc,yc])
        length =(end - begin)
        begin -=length/20
        end += length/20
        pa = center + r * np.array([np.cos(begin),np.sin(begin)])
        pb = center + r * np.array([np.cos(end),np.sin(end)])
        lines=np.array([pa,center,pb])
        plt.plot(lines[:,0], lines[:,1])
        ax.axis('equal')
        # Show the plot
        # plt.show()
        print('saved to cam_array_explain.png')
        fig.savefig("cam_array_explain.png")



    jsonstr=json.dumps({'transform':transform.tolist(),'circle':(xc,yc,r),'phi_range':(begin,end), 'z_range':zrange})
    if special_mode:
        return jsonstr
    if cut_range_output_file is not None:
        with open(cut_range_output_file,'w') as f:
            f.write(jsonstr)
            print(f'written file {cut_range_output_file}')
    else:
        print(jsonstr)

import sys
if __name__ == '__main__':
    # usage()
    try:
        #[--verbose|-v] file_of_cam.json [cut_range_output_file]
        verbose=False
        file_of_cam, cut_range_output_file = '/home/zhangzeheng/git/nerf2mesh/poses.txt' ,None
        args =sys.argv[1:]
        if len(args)==0:
            # usage()
            exit()
        if '--verbose' in  args or '-v' in  args:
            verbose = True
            args = args[1:]
        if len(args) >0:file_of_cam = args[0]
        if len(args) >1:
            cut_range_output_file = args[1]
    except:
        usage()
        exit()
    main(cam_file = file_of_cam,verbose = verbose,  cut_range_output_file = cut_range_output_file)
        
