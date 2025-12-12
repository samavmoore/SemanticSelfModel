import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Lasso, Ridge
from pl_module import VSM, BlendedSDF, CorrespondenceVSM
import mujoco
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from skimage import measure
import point_cloud_utils as pcu


def get_template_centers():
    state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
    state = state[7:]
    sdf0 = get_sdf_stage_0("/home/sammoore/Downloads/split_siren/", joint_pos=state)
    v, f, n, values = measure.marching_cubes(sdf0, level=0.0)

    target_radius = 15  # 1% of the bounding box radius

    # Generate barycentric coordinates of random samples
    fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=-1, radius=target_radius)

    # Interpolate the vertex positions and normals using the returned barycentric coordinates
    # to get sample positions and normals
    rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, n)

    # rescalse the vertices back to the original grid
    rand_positions = rescale_vertices(rand_positions, (-0.65, 0.65), 200)

    # calculate the distance between each point 
    dists = np.zeros((rand_positions.shape[0], rand_positions.shape[0]))
    for i in range(rand_positions.shape[0]):
        for j in range(rand_positions.shape[0]):
            dists[i, j] = np.linalg.norm(rand_positions[i, :] - rand_positions[j, :])
            if i == j:
                dists[i, j] = np.inf
    
    # get the minimum distance between each point
    min_dists = np.min(dists, axis=1)

    return rand_positions, min_dists

def get_clusters(ground_points, centers, descriptor_tracker, pose, threshold=0.195):
    n_clusters = centers.shape[0] + 1

    # concate the pose with the centers
    centers = np.concatenate((centers, pose.repeat(centers.shape[0], axis=0)), axis=1)

    # track the center given the pose
    new_centers = descriptor_tracker(torch.from_numpy(centers).float().to('cuda')).cpu().detach().numpy()

    # get the distances between the ground points and the new centers
    distances = np.zeros((ground_points.shape[0], n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.linalg.norm(ground_points - new_centers[i, :], axis=1)




def get_sdf_stage_0(log_dir, grid_size=200, bound_box=(-.65, .65), joint_pos=None):
    model = VSM.load_from_checkpoint(f'{log_dir}/checkpoints/newest.ckpt').to('cuda')
    model.eval()
    #model.model.stage1()

    lb = bound_box[0]
    ub = bound_box[1]
    x = np.linspace(lb, ub, num=grid_size)
    y = np.linspace(lb, ub, num=grid_size)
    z = np.linspace(lb, ub, num=grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


    state = joint_pos
    mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
    mjdata = mujoco.MjData(mjmodel)
    joint_limits = mjmodel.jnt_range[1:, :]
    state = 2*(state - joint_limits[:, 0])/(joint_limits[:, 1] - joint_limits[:, 0]) - 1
    pose = state
    state = np.repeat(state[np.newaxis, :], points.shape[0], axis=0)
    states = torch.from_numpy(state).float().to('cuda')
    #model = SwitchingSDFModule.load_from_checkpoint(f'/home/sammoore/Documents/PointCloud-PointForce/lightning_logs/version_156/checkpoints/newest.ckpt').to('cuda')
    #siren = model.sdf_net
    #switch = model.blend_net
    input = torch.cat((torch.from_numpy(points).float().to('cuda'), states), dim=1)

    batch_size = 100000  # Reduced batch size
    sdf = []

    with torch.no_grad():
        model.eval()
        #siren.eval()
        #switch.eval()
        for i in range(0, points.shape[0], batch_size):
            input0 = input[i:i+batch_size]
            sdf0 = model.model(input0)
            #sdf0 = siren(input0
            #switch0 = torch.sigmoid(switch(input0) - 6)
            #switch0 = switch(input0)
            #sdf0 = torch.mul(switch0, sdf0) + torch.mul((1-switch0), input0[:, :3].norm(dim=1).unsqueeze(-1))
            sdf.append(sdf0.cpu())

    sdf = torch.cat(sdf)

    sdf = sdf.reshape((grid_size, grid_size, grid_size))
    sdf = sdf.cpu().detach().numpy()
    return sdf

def load_data():
    path = "./dynamics_data/"
    # load all files in path
    files = os.listdir(path)
    files = [path + file for file in files]
    # concatenate all files into one array
    for i, file in enumerate(files):
        if file != "./dynamics_data/joint_torques.csv":
            #print(file)
            data_load = np.loadtxt(f"{file}",
                    delimiter=",")
            if len(data_load.shape) == 1:
                data_load = data_load[1:]
                data_load = data_load.reshape(data_load.shape[0], 1)
            else:
                data_load = data_load[1:, :]

            if i == 0:
                data = data_load
            else:
                data = np.concatenate((data, data_load), axis=1)
        
    states = data
    inputs = np.loadtxt(f"{path}joint_torques.csv", delimiter=",")
    indx_dict ={"joint_pos": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "body_quat": [12, 13, 14, 15],
                "body_height": [16],
                "body_vel": [17, 18, 19],
                "body_ang_vel": [20, 21, 22],
                "joint_vel": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]}
    
    return states, inputs, indx_dict



def get_features(data, delay_steps, n_principal_components):
    features = get_main_features(data, delay_steps)
    # get principal components of delayed states (n_features:)
    # first standardize the data
    _, n_features = data.shape
    og_states = features[:, :n_features]
    data = features
    features, principal_components = get_pc_features(data, n_principal_components)

    # make the features [cos(x), sin(x)] for each feature
    #features = np.concatenate((np.cos(features), np.sin(features)), axis=1)
    #print(features.shape)
    # make features [x, x^2, x^3, ...]
    features = np.concatenate((features, features**2, features**3), axis=1)
    # normalize the features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = features - mean
    features = features / std 
    # plot the first two principal components as a time series
    #print(features.shape)
    plt.plot(features[:100, 0], label="pc0")
    plt.plot(features[:100, 1], label="pc1")
    plt.plot(features[:100, n_principal_components], label="pc0^2")
    plt.plot(features[:100, n_principal_components + 1], label="pc1^2")
    plt.plot(features[:100, 2*n_principal_components], label="pc0^3")
    plt.plot(features[:100, 2*n_principal_components + 1], label="pc1^3")
    plt.legend()
    plt.savefig("poly_principal_components.png")
    plt.close()

    return og_states, features, principal_components, mean, std
    

def get_pc_features(data, n_principal_components):
    data = data - np.mean(data, axis=0)
    data = data / np.std(data, axis=0)
    # get covariance matrix
    cov = np.cov(data.T)
    # get eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov)
    # sort eigenvectors by eigenvalues
    idx = np.argsort(eig_vals)[::-1]
    # plot eigenvalues to see how many to keep
    # plt.plot(eig_vals)
    # # plot cumulative sum
    #plt.xlim(0, 40)
    #plt.savefig("eigenvalues.png")
    plt.plot(np.cumsum(eig_vals)/np.sum(eig_vals))
    plt.xlim(0, 350)
    plt.savefig("cumsum_eigenvalues.png")
    plt.close()
    #cum_sum = np.cumsum(eig_vals)/np.sum(eig_vals)
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    # get principal components
    principal_components = eig_vecs[:, :n_principal_components]
    # get features
    features = np.dot(data, principal_components)

    # plot the first two principal components as a time series
    plt.plot(features[:50, 0])
    plt.plot(features[:50, 1])
    plt.savefig("principal_components.png")
    plt.close()
    return features, principal_components


def get_main_features(states, delay_steps):
    # states: (n_samples, n_features)
    # want the output to be (n_samples - delay_steps, n_features*(delay_steps + 1))
    n_samples, n_features = states.shape
    n_features_out = n_features * (delay_steps + 1)
    n_samples_out = n_samples - delay_steps
    features = np.zeros((n_samples_out, n_features_out))
    state_t = states[delay_steps:, :]
    features[:, :n_features] = state_t
    for i in range(n_samples_out):
        prev_states = states[i:i+delay_steps, :]
        #prev_states = prev_states[::5, :]
        features[i, n_features:] = prev_states.flatten()
    
    #print(features.shape)
    return features


def get_ground_position_in_body_frame(body_height, body_quat, grid, vox_res=0.01):
    """
    Determine the position of the ground in the robot's body frame using a voxel grid, automatically calculating voxel resolution.
    
    Parameters:
    - body_height: numpy array of shape (n_samples,), heights of the robot base above the ground
    - body_quat: numpy array of shape (n_samples, 4), quaternions representing the orientation of the robot
    - grid: numpy array of shape (n_grid_pts, 3), points in the voxel grid relative to the body's origin
    
    Returns:
    - ground_positions: numpy array of shape (n_samples, n_grid_pts), binary grid where 1 indicates ground presence
    """
    n_samples = body_height.shape[0]
    n_grid_pts = grid.shape[0]
    ground_positions = np.zeros((n_samples, n_grid_pts), dtype=int)  # Using int for binary data

    # Calculate voxel resolution (assuming the grid is regularly spaced along at least one axis)
    voxel_resolution = vox_res
    for i in range(n_samples):
        # Get the rotation matrix from the quaternion
        r = R.from_quat(body_quat[i, :])
        rot_mat = r.as_matrix()
        #translation = np.array([0, 0, body_height[i, 0]]).repeat(n_grid_pts).reshape(-1, 3)
        # Transform each grid point
        print(body_height[i, 0])
        transformed_grid = np.dot(grid, rot_mat.T) + np.array([0, 0, body_height[i, 0]])

        # Determine which points are on the ground using half voxel resolution as threshold
        ground_threshold = voxel_resolution / 2
        is_ground = np.abs(transformed_grid[:, 2]) < ground_threshold

        # Store the result in the ground_positions matrix
        ground_positions[i, :] = is_ground.astype(int)

    return ground_positions


def load_dense_self_models():
    log_dir = '/home/sammoore/Documents/PointCloud-PointForce/stage_1_logs/siren_split_trial_0'
    correspondences_log_dir="/home/sammoore/Documents/PointCloud-PointForce/correspondence_logs/trial_0"
    sdf_model = BlendedSDF.load_from_checkpoint(f'{log_dir}/checkpoints/newest.ckpt').to('cuda')
    sdf_model.eval()
    corr_net_base = CorrespondenceVSM.load_from_checkpoint(f"{correspondences_log_dir}/checkpoints/newest.ckpt").to('cuda')
    descriptor_lookup = corr_net_base.model
    descriptor_lookup.eval()
    descriptor_tracker = corr_net_base.model_inv
    descriptor_tracker.eval()
    return sdf_model, descriptor_lookup, descriptor_tracker

def get_contact_features(data):
    pass

def create_grid(bound_box=(-0.65, 0.65), grid_size=100):
    lb = bound_box[0]
    ub = bound_box[1]
    x = np.linspace(lb, ub, num=grid_size)
    y = np.linspace(lb, ub, num=grid_size)
    z = np.linspace(lb, ub, num=grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    return points

def rescale_vertices(vertices, bound_box, grid_size):
    # Calculate the physical length of each voxel along each axis
    voxel_size = (bound_box[1] - bound_box[0]) / grid_size
    # Rescale vertices from grid indices to physical coordinates
    vertices = vertices * voxel_size + bound_box[0]
    return vertices

if __name__ == "__main__":
    rand_positions, min_dists = get_template_centers()
    plt.hist(min_dists, bins=100)
    plt.savefig("min_dists.png")

    # states, inputs, idx_dict = load_data()

    # n_pcs = 500
    # states, feats, pc, mean, std = get_features(states, 20, n_pcs)

    # # define the models
    
    # #sdf, descriptor_lookup, descriptor_tracker = load_dense_self_models()
    # # get the grid
    # grid = create_grid(grid_size=70)
    # # get the ground positions (testing just the first 2 samples)
    # data_idx = np.random.randint(0, states.shape[0])
    # body_height = states[data_idx:data_idx+1 ,idx_dict["body_height"]]
    # body_quat = states[data_idx:data_idx+1, idx_dict["body_quat"]]
    # ground_positions = get_ground_position_in_body_frame(body_height, body_quat, grid, vox_res=1.3/70)
    # # ground xyz points
    # print(ground_positions.shape)
    # # number of ground points
    # print(np.sum(ground_positions[0, :]))
    # ground_xyz = grid[ground_positions[0, :] == 1]
    # # get the sdf
    # sdf0 = get_sdf_stage_0("/home/sammoore/Downloads/split_siren/", joint_pos=states[data_idx, idx_dict["joint_pos"]])
    # v, f, n, values = measure.marching_cubes(sdf0, level=0.0)

    # target_num_pts= 20

    # # Generate barycentric coordinates of random samples
    # fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=target_num_pts)

    # # Interpolate the vertex positions and normals using the returned barycentric coordinates
    # # to get sample positions and normals
    # rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    # rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, n)

    # # rescalse the vertices back to the original grid
    # vertices = rescale_vertices(rand_positions, (-0.65, 0.65), 200)


    # # plot the ground points and the sdf with open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(ground_xyz)
    # sdf_pcd = o3d.geometry.PointCloud()
    # sdf_pcd.points = o3d.utility.Vector3dVector(vertices)
    # sdf_pcd.colors = o3d.utility.Vector3dVector(np.array([0, 0, 0]).repeat(vertices.shape[0]).reshape(-1, 3))

    # # full sdf vertices but shifted to the right
    # sdf_pcd_shifted = o3d.geometry.PointCloud()
    # v = rescale_vertices(v, (-0.65, 0.65), 200)
    # sdf_pcd_shifted.points = o3d.utility.Vector3dVector(v + np.array([1, 1, 1]))
    # sdf_pcd_shifted.colors = o3d.utility.Vector3dVector(np.array([0, 0, 0]).repeat(v.shape[0]).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd, sdf_pcd, sdf_pcd_shifted])
    # plot the sdf

    # state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
    # state = state[7:]
    # sdf0 = get_sdf_stage_0("/home/sammoore/Downloads/split_siren/", joint_pos=state)
    # v, f, n, values = measure.marching_cubes(sdf0, level=0.0)

    # target_num_pts= 20

    # # Generate barycentric coordinates of random samples
    # fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=target_num_pts)

    # # Interpolate the vertex positions and normals using the returned barycentric coordinates
    # # to get sample positions and normals
    # rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    # rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, n)

    # target_radius = 15  # 1% of the bounding box radius

    # # Generate barycentric coordinates of random samples
    # fid, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=-1, radius=target_radius)

    # # Interpolate the vertex positions and normals using the returned barycentric coordinates
    # # to get sample positions and normals
    # rand_positions = pcu.interpolate_barycentric_coords(f, fid, bc, v)
    # rand_normals = pcu.interpolate_barycentric_coords(f, fid, bc, n)

    # # rescalse the vertices back to the original grid
    # rand_positions = rescale_vertices(rand_positions, (-0.65, 0.65), 200)
    # v = rescale_vertices(v, (-0.65, 0.65), 200)

    # # rescale the target radius
    # target_radius = rescale_vertices(np.array([[target_radius, target_radius, target_radius]]), (-0.65, 0.65), 200)[0, 0]
    # print(target_radius)
    # print(rand_positions.shape)
    # sdf_pcd = o3d.geometry.PointCloud()
    # sdf_pcd.points = o3d.utility.Vector3dVector(rand_positions)
    # sdf_pcd.colors = o3d.utility.Vector3dVector(np.array([0, 0, 0]).repeat(rand_positions.shape[0]).reshape(-1, 3))
    # full_sdf_pcd = o3d.geometry.PointCloud()
    # full_sdf_pcd.points = o3d.utility.Vector3dVector(v + np.array([0, 0.65, 0]))
    # full_sdf_pcd.colors = o3d.utility.Vector3dVector(np.array([0, 0, 0]).repeat(v.shape[0]).reshape(-1, 3))
    # o3d.visualization.draw_geometries([sdf_pcd, full_sdf_pcd])
    # # normalize the states
    # states = states - np.mean(states, axis=0)
    # states = states / np.std(states, axis=0)
    # # reconstruct the states using the principal components and features[:, :250]
    # unscaled_feats = feats * std
    # unscaled_feats = unscaled_feats + mean
    # reconstructed_states = np.dot(unscaled_feats[:, :n_pcs], pc.T)
    # #print(reconstructed_states.shape)
    # #print(states.shape)
    # plt.scatter(states[:, 17], reconstructed_states[:, 17])
    # plt.savefig("reconstructed_states_0.png")
    # plt.close()

    # plt.scatter(states[:, 1], reconstructed_states[:, 1])
    # plt.savefig("reconstructed_states_1.png")
    # plt.close()

    # plt.scatter(states[:, 2], reconstructed_states[:, 2])
    # plt.savefig("reconstructed_states_2.png")
    # plt.close()

    # plt.scatter(states[:, 3], reconstructed_states[:, 3])
    # plt.savefig("reconstructed_states_3.png")
    # plt.close()

    # plt.scatter(states[:, 4], reconstructed_states[:, 4])
    # plt.savefig("reconstructed_states_4.png")
    # plt.close()
    
    # plt.scatter(states[:, 5], reconstructed_states[:, 5])
    # plt.savefig("reconstructed_states_5.png")
    # plt.close()

    # plt.scatter(states[:, 6], reconstructed_states[:, 6])
    # plt.savefig("reconstructed_states_6.png")
    # plt.close()
