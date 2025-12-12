import mujoco
import numpy as np
from Utils.cam_utils import Camera, generate_axes
import open3d as o3d
import os
import time
import yaml
import point_cloud_utils as pcu
import json

class CollectVSMData:
    def __init__(self, vsm_configs_yaml):
        # Load configurations from a YAML file
        try:
            with open(vsm_configs_yaml, 'r') as file:
                self.vsm_configs = yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load or parse the YAML file: {e}")

        # Initialize simulation components from the loaded configuration
        self.mjmodel = mujoco.MjModel.from_xml_path(self.vsm_configs['model_path'])
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.mjrenderer = mujoco.Renderer(self.mjmodel, self.vsm_configs['pix_height'], self.vsm_configs['pix_width'])

        self.n_samples = self.vsm_configs['n_samples'] 
        # Setup cameras based on the configurations
        self.cams = self.setup_cameras(self.vsm_configs['cam_configs'])
        self.vol_bbox = self.vsm_configs['vol_bbox']
        self.seed = self.vsm_configs['seed']
        self.data_dir = self.vsm_configs['data_dir']
        # set random seed
        np.random.seed(self.seed)
        self.starting_sample = None

        # if the data directory does not exist, create it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


    def setup_cameras(self, cam_configs):
        cams = []
        for cam_config in cam_configs:
            name = cam_config['name']
            lookat = cam_config['lookat']
            distance = cam_config['distance']
            azimuth = cam_config['azimuth']
            elevation = cam_config['elevation']
            cam = Camera(name, self.mjmodel, self.mjdata, self.mjrenderer, lookat, distance, azimuth, elevation)
            cams.append(cam)
        return cams
    
    def collect_data(self, start_sample=0, jnt_by_jnt=False, n_steps=100):
        """
        Collect data from the simulation.

        Parameters:
        - start_sample: The starting index for sampling.
        - jnt_by_jnt: Boolean flag to sample each joint independently.
        - n_steps: Number of steps to interpolate each joint's range when jnt_by_jnt is True.
        """
        joint_limits = self.mjmodel.jnt_range[1:, :]
        jnt_qposadr = self.mjmodel.jnt_qposadr[1:]
        sample = start_sample
        self.starting_sample = start_sample
        start_time = time.time()

        if jnt_by_jnt:
            # If joint-by-joint mode, define a step for each joint separately
            angle_steps = [np.linspace(joint_limits[j, 0], joint_limits[j, 1], n_steps) for j in range(joint_limits.shape[0])]
            n_samples = sum(len(steps) for steps in angle_steps)
        else:
            # Otherwise, use a defined number of samples
            n_samples = self.n_samples

        jnt_data = np.zeros((n_samples, self.mjmodel.njnt-1))

        if jnt_by_jnt:
            # this just runs a for loop through all the joint angles 
            for joint_idx, angles in enumerate(angle_steps):
                for angle in angles:
                    nominal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
                    nominal += np.random.randn(nominal.shape[0])*55*(np.pi/180)
                    nominal[:7] = 0 
                    nominal[10] = 0 + np.random.randn()*30*(np.pi/180)
                    nominal[13] = 0 + np.random.randn()*30*(np.pi/180)
                    nominal[16] = 0 + np.random.randn()*30*(np.pi/180)
                    self.mjdata.qpos[:] =  nominal  #np.zeros(self.mjmodel.nq)
                    #self.mjdata.qpos[jnt_qposadr[joint_idx]] = angle
                    mujoco.mj_step(self.mjmodel, self.mjdata)
                    if self.mjdata.ncon > 0:
                        continue  # Skip this sample due to self-collision
                    jnt_data[sample-start_sample, :] = self.mjdata.qpos[jnt_qposadr]
                    body_pcd, legs_pcd, off_pcd, off_d = self.get_scene()
                    #o3d.visualization.draw_geometries([on_pcd])
                    #o3d.visualization.draw_geometries([off_pcd])
                    self.save_data(jnt_data, body_pcd, legs_pcd, off_pcd, off_d, sample)
                    if sample % 50 == 0:
                        print(f"Collecting sample {sample}")
                        print(f"Time elapsed: {time.time() - start_time}")

                    sample += 1

        else:
            while sample < n_samples+start_sample:
                qpos = np.random.uniform(joint_limits[:9, 0], joint_limits[:9, 1])
                self.mjdata.qpos[:] = np.zeros(self.mjmodel.nq)
                self.mjdata.qpos[jnt_qposadr[:9]] = qpos
                mujoco.mj_step(self.mjmodel, self.mjdata)
                if self.mjdata.ncon > 0:
                    continue  # Skip this sample due to self-collision
                jnt_data[sample-start_sample, :] = self.mjdata.qpos[jnt_qposadr]
                body_pcd, legs_pcd, off_pcd, off_d = self.get_scene()
                #o3d.visualization.draw_geometries([on_pcd])
                #o3d.visualization.draw_geometries([off_pcd])
                self.save_data(jnt_data, body_pcd, legs_pcd, off_pcd, off_d, sample)

                if sample % 50 == 0:
                    print(f"Collecting sample {sample}")
                    print(f"Time elapsed: {time.time() - start_time}")

                sample += 1
            

    def get_scene(self):
        # Handling point cloud processing and saving, extracted from main loop
        for i, cam in enumerate(self.cams):
            valid_points, color_map = cam.get_segmented_pointcloud(bounds=self.vol_bbox, exclude_geom_id=[0, 1, 2, 3, 4, 5])
            #valid_points = cam.get_pointcloud(bounds=self.vol_bbox)
            point_cloud_o3d = valid_points if i == 0 else point_cloud_o3d + valid_points
        
        point_cloud_o3d.estimate_normals(fast_normal_computation=False)
        point_cloud_o3d.orient_normals_consistent_tangent_plane(3)
        point_cloud_o3d.rotate(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))
        o3d.visualization.draw_geometries([point_cloud_o3d])

        # convert to numpy array
        p = np.asarray(point_cloud_o3d.points)
        n = np.asarray(point_cloud_o3d.normals)
        body_idx = np.where(np.all(point_cloud_o3d.colors == np.array([1, 0, 0]), axis=1))
        legs_idx = np.where(np.all(point_cloud_o3d.colors == np.array([0, 0, 1]), axis=1))


        p_body = p[body_idx]
        p_legs = p[legs_idx]
        n_body = n[body_idx]
        n_legs = n[legs_idx]
        pc_body = o3d.geometry.PointCloud()
        pc_body.points = o3d.utility.Vector3dVector(p_body)
        pc_body.normals = o3d.utility.Vector3dVector(n_body)
        pc_legs = o3d.geometry.PointCloud()
        pc_legs.points = o3d.utility.Vector3dVector(p_legs)
        pc_legs.normals = o3d.utility.Vector3dVector(n_legs)
        #o3d.visualization.draw_geometries([pc_legs])
        #o3d.visualization.draw_geometries([pc_body])
        
        p = np.asarray(point_cloud_o3d.points)
        n = np.asarray(point_cloud_o3d.normals)
        surfel_rad = np.ones(p.shape[0], dtype=p.dtype) * 0.002

        # A triangle mesh representing surfel geometry
        v, f = pcu.pointcloud_surfel_geometry(p, n, surfel_rad)
        #v, f = pcu.make_mesh_watertight(v, f)
        n = pcu.estimate_mesh_vertex_normals(v, f)
        #pc = o3d.geometry.PointCloud()
        #pc.points = o3d.utility.Vector3dVector(v)
        #pc.normals = o3d.utility.Vector3dVector(n)
        #o3d.visualization.draw_geometries([pc])

        #pc = point_cloud_o3d.voxel_down_sample(voxel_size=0.1) # just for visualization
        #d = np.random.uniform(0, 1, len(pc.points))
        #query_points = np.asarray(pc.points) + d[:, np.newaxis] * np.asarray(pc.normals)
        q_x = np.random.normal(0, 0.6, 55000)
        q_y = np.random.normal(0, 0.4, 55000)
        q_z = np.random.normal(-0.2, 0.5, 55000)
        query_points = np.stack((q_x, q_y, q_z), axis=-1)
        sdf, fid, bc = pcu.signed_distance_to_mesh(query_points, v, f)
        # get rid of query points inside the mesh
        #print(f"Number of query points: {len(query_points)}")
        query_points = query_points[sdf > 0]
        #print(f"Number of query points outside the mesh: {len(query_points)}")
        dists, fid, bc = pcu.closest_points_on_mesh(query_points, v, f)
        closest_points = pcu.interpolate_barycentric_coords(f, fid, bc, v)
        query_points_pc = o3d.geometry.PointCloud()
        query_points_pc.points = o3d.utility.Vector3dVector(query_points)
        closest_points_pc = o3d.geometry.PointCloud()
        closest_points_pc.points = o3d.utility.Vector3dVector(closest_points)

        difference_vector = closest_points - query_points
        norm = np.linalg.norm(difference_vector, axis=1)
        # Compute the gradient of the signed distance function
        grad = -difference_vector / norm[:, np.newaxis]
        query_points_pc.normals = o3d.utility.Vector3dVector(grad)
        #robot_pc = o3d.geometry.PointCloud()
        #robot_pc.points = o3d.utility.Vector3dVector(v)
        # Visualize with normals
        #o3d.visualization.draw_geometries([query_points_pc, robot_pc], point_show_normal=True)
        # Verify that closest_points = query_points - grad * sdf
        assert np.allclose(closest_points, query_points - grad * norm[:, np.newaxis])
        assert np.allclose(np.linalg.norm(grad, axis=1), 1.0)
        # randomly downsample the point clouds


        #idx_pc = np.random.choice(np.arange(len(pc.points)), 50000)
        #pc = pc.select_by_index(idx_pc)
        idx_pc_body = np.random.choice(np.arange(len(pc_body.points)), 5000)
        pc_body = pc_body.select_by_index(idx_pc_body)
        #o3d.visualization.draw_geometries([pc_body])
        idx_pc_legs = np.random.choice(np.arange(len(pc_legs.points)), 45000)
        pc_legs = pc_legs.select_by_index(idx_pc_legs)
        #o3d.visualization.draw_geometries([pc_legs])

        # off surface point cloud
        idx_query = np.random.choice(np.arange(len(query_points)), 50000)
        query_points_pc = query_points_pc.select_by_index(idx_query)
        norm = norm[idx_query]
        return pc_body, pc_legs, query_points_pc, norm
    
    def save_data(self, jnt_data, body_pc, legs_pc, off_pc, off_d, sample):
        o3d.io.write_point_cloud(f"{self.data_dir}/body_pc_{sample}.ply", body_pc)
        o3d.io.write_point_cloud(f"{self.data_dir}/legs_pc_{sample}.ply", legs_pc)
        o3d.io.write_point_cloud(f"{self.data_dir}/off_surface_pc_{sample}.ply", off_pc)
        np.save(f"{self.data_dir}/off_surface_d_{sample}.npy", off_d)
        np.save(f"{self.data_dir}/joint_data_{sample}.npy", jnt_data[sample-self.starting_sample, :])



class CollectVSMSemanticData:
    def __init__(self, vsm_configs_yaml):
        # Load configurations from a YAML file
        try:
            with open(vsm_configs_yaml, 'r') as file:
                self.vsm_configs = yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Failed to load or parse the YAML file: {e}")

        # Initialize simulation components from the loaded configuration
        self.mjmodel = mujoco.MjModel.from_xml_path(self.vsm_configs['model_path'])
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.mjrenderer = mujoco.Renderer(self.mjmodel, self.vsm_configs['pix_height'], self.vsm_configs['pix_width'])

        self.n_samples = self.vsm_configs['n_samples'] 
        # Setup cameras based on the configurations
        self.cams = self.setup_cameras(self.vsm_configs['cam_configs'])
        self.vol_bbox = self.vsm_configs['vol_bbox']
        self.seed = self.vsm_configs['seed']
        self.data_dir = self.vsm_configs['data_dir']
        # set random seed
        np.random.seed(self.seed)
        self.starting_sample = None

        # if the data directory does not exist, create it
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)


    def setup_cameras(self, cam_configs):
        cams = []
        for cam_config in cam_configs:
            name = cam_config['name']
            lookat = cam_config['lookat']
            distance = cam_config['distance']
            azimuth = cam_config['azimuth']
            elevation = cam_config['elevation']
            cam = Camera(name, self.mjmodel, self.mjdata, self.mjrenderer, lookat, distance, azimuth, elevation)
            cams.append(cam)
        return cams
    
    def collect_data(self, start_sample=0):
        """
        Collect data from the simulation.

        Parameters:
        - start_sample: The starting index for sampling.
        - jnt_by_jnt: Boolean flag to sample each joint independently.
        - n_steps: Number of steps to interpolate each joint's range when jnt_by_jnt is True.
        """
        joint_limits = self.mjmodel.jnt_range[1:, :]
        jnt_qposadr = self.mjmodel.jnt_qposadr[1:]
        sample = start_sample
        self.starting_sample = start_sample
        start_time = time.time()


        n_samples = self.n_samples

        jnt_data = np.zeros((n_samples, self.mjmodel.njnt-1))

        while sample < n_samples+start_sample:
            qpos = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
            #qpos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
            z = np.random.randn(qpos.shape[0])
            while np.any(z > 1) or np.any(z < -1):
                z = np.random.randn(qpos.shape[0])

            self.mjdata.qpos[:] = np.zeros(self.mjmodel.nq)
            self.mjdata.qpos[jnt_qposadr[:]] = qpos
            mujoco.mj_step(self.mjmodel, self.mjdata)
            if self.mjdata.ncon > 0:
                continue  # Skip this sample due to self-collision
            jnt_data[sample-start_sample, :] = self.mjdata.qpos[jnt_qposadr]
            #fl_hip_jnt = self.mjmodel.joint('FL_hip_joint')
            #fl_hip_body = self.mjmodel.body('FL_hip')
            #jnt_pos = self.mjdata.qpos[jnt_qposadr]
            #mj_rot = self.mjdata.xmat[fl_hip_body.id]
            body_pcd, color_map = self.get_scene()
            #o3d.visualization.draw_geometries([on_pcd])
            #o3d.visualization.draw_geometries([off_pcd])
            self.save_data(jnt_data, body_pcd, sample, color_map)

            if sample % 50 == 0:
                print(f"Collecting sample {sample}")
                print(f"Time elapsed: {time.time() - start_time}")

            sample += 1
            

    def get_scene(self):
        # Handling point cloud processing and saving, extracted from main loop
        for i, cam in enumerate(self.cams):
            valid_points, color_map = cam.get_segmented_pointcloud(bounds=self.vol_bbox, exclude_geom_id=[0, 1, 2, 3, 4, 5], groups='go2')
            point_cloud_o3d = valid_points if i == 0 else point_cloud_o3d + valid_points

        point_cloud_o3d, idx = point_cloud_o3d.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.3) #1.3, 2.5
        point_cloud_o3d.rotate(np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), center=(0, 0, 0))

        # convert to numpy array
        p = np.asarray(point_cloud_o3d.points)

        body_idx = np.where(np.all(point_cloud_o3d.colors == np.array([0, 0, 0]), axis=1))
        p_body = p[body_idx]
        pc_body = o3d.geometry.PointCloud()
        pc_body.points = o3d.utility.Vector3dVector(p_body)
        pc_body.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 0]), (len(p_body), 1)))

        idx_pc_body = np.random.choice(np.arange(len(pc_body.points)), 5000)
        pc_body = pc_body.select_by_index(idx_pc_body)

        rigid_links = np.unique(point_cloud_o3d.colors, axis=0)

        # balancing the number of points in each rigid link
        for link in rigid_links:
            if np.all(link == np.array([0, 0, 0])):
                continue
            link_idx = np.where(np.all(point_cloud_o3d.colors == link, axis=1))
            p_link = p[link_idx]

            pc_link = o3d.geometry.PointCloud()
            pc_link.points = o3d.utility.Vector3dVector(p_link)
            pc_link.colors = o3d.utility.Vector3dVector(np.tile(link, (len(p_link), 1)))
            idx_pc_link = np.random.choice(np.arange(len(pc_link.points)), 5000)
            pc_link = pc_link.select_by_index(idx_pc_link)

            pc_body += pc_link

        return pc_body, color_map
    
    def save_data(self, jnt_data, body_pc, sample, color_map):
        #print("color_map", color_map)
        o3d.io.write_point_cloud(f"{self.data_dir}/pc_{sample}.ply", body_pc)

        if sample == 0:
            # save color map as a json file
            with open(f"{self.data_dir}/color_map.json", 'w') as f:
                json.dump(color_map, f)

        np.save(f"{self.data_dir}/joint_data_{sample}.npy", jnt_data[sample-self.starting_sample, :])



if __name__ == "__main__":


    import os
    os.environ['MUJOCO_GL'] = 'egl'
    go2_configs = "./Configs/go2_vsm_configs_test.yaml"
    #vsm_data = CollectVSMData(go2_configs)
    vsm_data = CollectVSMSemanticData(go2_configs)

    t0 = time.time()
    vsm_data.collect_data(0) #62450 #1900
    print(f"Collecting data took {(time.time() - t0)/vsm_data.n_samples} seconds per sample.")
    
    #data_dir = "./Data/go2_vsm"
    #data_dir "/home/sammoore/Documents/PointCloud-PointForce/Debug_Data"
    #load all data ending in npy as txt files and convert to npy and overwrite
    #for file in os.listdir(data_dir):
    #    if file.endswith(".npy"):
    #        data = np.loadtxt(f"{data_dir}/{file}")
    #        np.save(f"{data_dir}/{file}", data)

            