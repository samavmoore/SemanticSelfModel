import mujoco
import numpy as np
import open3d as o3d
import mediapy as media
import matplotlib.pyplot as plot


def depth_to_pointcloud(depth_image, fx, fy, cx, cy, bounds=None, depth_scale=1.0, segmentation_image=None, geom_id_exclude=None, groups=None):
    """
    Convert a depth image to a point cloud.

    Parameters:
    - depth_image: 2D numpy array of depth values.
    - fx, fy: Focal lengths of the camera in pixels.
    - cx, cy: Optical center of the camera in pixels.
    - bounds: Bounding box for the point cloud in the form [xmin, xmax, ymin, ymax, zmin, zmax].
    - depth_scale: Scale to convert depth units to meters.

    Returns:
    - open3d.geometry.PointCloud object
    """
    # Get the shape of the depth image
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(x, y)

    # Convert from pixel coordinates to camera coordinates
    Z = depth_image / depth_scale
    X = (px - cx) * Z / fx
    Y = (py - cy) * Z / fy

    # Stack coordinates into (N, 3) array
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    # valid_points = points[Z.ravel() > 0, :]
    # if bounds is not None:
    #     valid_x = (valid_points[:, 0] > bounds[0]) & (valid_points[:, 0] < bounds[1])
    #     valid_points = valid_points[valid_x]
    #     valid_y = (valid_points[:, 1] > bounds[2]) & (valid_points[:, 1] < bounds[3])
    #     valid_points = valid_points[valid_y]
    #     valid_z = (valid_points[:, 2] > bounds[4]) & (valid_points[:, 2] < bounds[5])
    #     valid_points = valid_points[valid_z]

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(valid_points)
    # o3d.visualization.draw_geometries([point_cloud])
    # Remove points with no depth
    #valid_points = points[Z.ravel() > 0, :]

    if groups == 'go2':
        #groups = {'Body': [-1], 'BL_Hip': [44, 45], 'BR_Hip': [32, 33], 'FL_Hip': [20, 21], 'FR_Hip': [8, 9], 'BL_Thigh': [47, 48], 'BR_Thigh': [35, 36], 'FL_Thigh': [23, 24], 'FR_Thigh': [11, 12], 'BL_Shank': [50, 51, 54], 'BR_Shank': [38, 39, 42], 'FL_Shank': [26, 27, 30], 'FR_Shank': [14, 15, 18]}
        groups = {'Body': [-1], 'RR_hip': [44, 45], 'RL_hip': [32, 33], 'FR_hip': [20, 21], 'FL_hip': [8, 9], 'RR_thigh': [47, 48], 'RL_thigh': [35, 36], 'FR_thigh': [23, 24], 'FL_thigh': [11, 12], 'RR_calf': [50, 51, 54], 'RL_calf': [38, 39, 42], 'FR_calf': [26, 27, 30], 'FL_calf': [14, 15, 18]}
        #color_map = {'Body': [0, 0, 0], 'BL_Hip': [0, 0, 1], 'BR_Hip': [1, 0, 0], 'FL_Hip': [0, 1, 0], 'FR_Hip': [1, 0, 1], 'BL_Thigh': [1, 1, 0], 'BR_Thigh': [0, 1, 1], 'FL_Thigh': [1, 1, 1], 'FR_Thigh': [.5, .5, .5], 'BL_Shank': [.5, 0, 0], 'BR_Shank': [0, .5, 0], 'FL_Shank': [0, 0, .5], 'FR_Shank': [.5, .5, 0]}
        color_map = {
                    'Body': [0, 0, 0],
                    'RR_hip': [1, 0, 0],
                    'RL_hip': [0, 0, 1],
                    'FR_hip': [1, 0, 1],
                    'FL_hip': [0, 1, 0],
                    'RR_thigh': [0, 1, 1],
                    'RL_thigh': [1, 1, 0],
                    'FR_thigh': [0.5, 0.5, 0.5],
                    'FL_thigh': [1, 1, 1],
                    'RR_calf': [0, 0.5, 0],
                    'RL_calf': [0.5, 0, 0],
                    'FR_calf': [0.5, 0.5, 0],
                    'FL_calf': [0, 0, 0.5]
                }
        color_to_id = {tuple(color): id for id, color in color_map.items()}
        #print(color_to_id)
        id_to_group = {id: group for group, ids in groups.items() for id in ids}

    else:
        color_map = {-1: [1, 0, 0], 0: [0, 0, 1]}

    # BL Hip: [44, 45]
    # BR Hip:  [32, 33]
    # FL Hip:  [20, 21]
    # FR Hip: [8, 9]
    # BL Thigh: [47, 48]
    # BR Thigh: [35, 36]
    # FL Thigh: [23, 24]
    # FR Thigh:[11, 12]
    # BL Shank: [50, 51, 54]
    # BR Shank: [38, 39, 42]
    # FL Shank: [26, 27, 30]
    # FR Shank: [14, 15, 18]

    # make color groups for each of the links above
    #groups = {'Body': [-1], 'BL Hip': [44, 45], 'BR Hip': [32, 33], 'FL Hip': [20, 21], 'FR Hip': [8, 9], 'BL Thigh': [47, 48], 'BR Thigh': [35, 36], 'FL Thigh': [23, 24], 'FR Thigh': [11, 12], 'BL Shank': [50, 51, 54], 'BR Shank': [38, 39, 42], 'FL Shank': [26, 27, 30], 'FR Shank': [14, 15, 18]}
    # create color map for each group us 0 .5 1

    #color_map = {'Body': [0, 0, 0], 'BL Hip': [0, 0, 1], 'BR Hip': [1, 0, 0], 'FL Hip': [0, 1, 0], 'FR Hip': [1, 0, 1], 'BL Thigh': [1, 1, 0], 'BR Thigh': [0, 1, 1], 'FL Thigh': [1, 1, 1], 'FR Thigh': [.5, .5, .5], 'BL Shank': [.5, 0, 0], 'BR Shank': [0, .5, 0], 'FL Shank': [0, 0, .5], 'FR Shank': [.5, .5, 0]}

    #id_to_group = {id: group for group, ids in groups.items() for id in ids}

    # if segmentation image is provided, filter out points that have the same id as geom_id
    if segmentation_image is not None:
        geom_ids = segmentation_image[:, :, 0]
        for i in range(len(geom_id_exclude)):
            geom_ids = np.where(geom_ids == geom_id_exclude[i], -1, geom_ids)
        if groups is None:
            geom_ids = np.where((geom_ids != -1), 0, geom_ids)
        #geom_ids = np.where((geom_ids != -1) & (geom_ids != 38) & (geom_ids != 39), 0, geom_ids)

        #valid_points = valid_points[geom_ids.ravel() >= 0, :]

        #colors = np.array([color_map[geom_ids.ravel()[i]] for i in range(len(geom_ids.ravel()))])
        #keys from values
        if groups is not None:
            colors = np.array([color_map[id_to_group[geom_ids.ravel()[i]]] for i in range(len(geom_ids.ravel()))])
        else:
            colors = np.array([color_map[geom_ids.ravel()[i]] for i in range(len(geom_ids.ravel()))])
    else:
        colors = np.ones_like(points) * [0, 0, 1]

    # Create a point cloud object

    valid_points = points[Z.ravel() > 0, :]
    valid_colors = colors[Z.ravel() > 0, :]

    #truncate the point cloud outside of the bounding box
    if bounds is not None:
        valid_x = (valid_points[:, 0] > bounds[0]) & (valid_points[:, 0] < bounds[1])
        valid_points = valid_points[valid_x]
        valid_y = (valid_points[:, 1] > bounds[2]) & (valid_points[:, 1] < bounds[3])
        valid_points = valid_points[valid_y]
        valid_z = (valid_points[:, 2] > bounds[4]) & (valid_points[:, 2] < bounds[5])
        valid_points = valid_points[valid_z]
        valid_colors = valid_colors[valid_x]
        valid_colors = valid_colors[valid_y]
        valid_colors = valid_colors[valid_z]


        
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)
    point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)

    # visualize the point cloud for debugging
    #o3d.visualization.draw_geometries([point_cloud])

    if groups is not None:
        return point_cloud, color_map
    else:
        return point_cloud

def generate_axes(scale):
    """
    Generate a set of lines representing the axes of a coordinate frame in Open3D.
    """
    points = [
        [0, 0, 0],
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, scale],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return [line_set, ]

class Camera(mujoco.MjvCamera):
    def __init__(self, name, model, data, renderer, lookat=None, distance=None, azimuth=None, elevation=None):
        super().__init__()
        self.model = model
        self.data = data
        self.renderer = renderer
        self.name = name
        self.I = None
        self.E = None
        self.update_camera(lookat, distance, azimuth, elevation)

    def update_camera(self, lookat=None, distance=None, azimuth=None, elevation=None):
        """Update the camera's position and orientation."""
        if lookat is not None:
            self.lookat = lookat
        if distance is not None:
            self.distance = distance
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            self.elevation = elevation
        self.update_camera_matrices()

    def update_camera_matrices(self):
        """Returns the Intrinsics and Extrinsic matrices of the camera."""
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. It is a stereo camera, so we average over
        # the left and right channels. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.
        self.renderer.update_scene(self.data, self)
        pos = np.mean([camera.pos for camera in self.renderer.scene.camera], axis=0)
        z = -np.mean([camera.forward for camera in self.renderer.scene.camera], axis=0)
        y = np.mean([camera.up for camera in self.renderer.scene.camera], axis=0)
        rot = np.vstack((np.cross(y, z), y, z))
        fov = self.model.vis.global_.fovy

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.renderer.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.renderer.width - 1) / 2.0
        image[1, 2] =  (self.renderer.height - 1) / 2.0 

        intrinsic = image @ focal

        extrinsic = rotation @ translation 

        self.I = intrinsic
        self.E = extrinsic
    
    def get_depth_meters(self):
        """Returns the depth image in meters."""
        self.renderer.update_scene(self.data, self)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth
    
    def get_depth_image(self):
        """Returns the depth image scaled so that is ready to be visualized as an image."""
        depth = self.get_depth_meters()
        # Shift nearest values to the origin.
        depth -= depth.min()
        # Scale by 2 mean distances of near rays.
        depth /= 2*depth[depth <= 1].mean()
        # Scale to [0, 255]
        pixels = 255*np.clip(depth, 0, 1)
        return pixels.astype(np.uint8)
    
    def get_rgb_image(self):
        """Returns the RGB image."""
        self.renderer.update_scene(self.data, self)
        rgb = self.renderer.render()
        return rgb
    
    def get_segmentation(self):
        self.renderer.update_scene(self.data, self)
        self.renderer.enable_segmentation_rendering()
        segmentation = self.renderer.render()
        self.renderer.disable_segmentation_rendering()
        return segmentation
    
    def get_segmentation_image(self):
        segmentation = self.get_segmentation()
        # Display the contents of the first channel, which contains object
        # IDs. The second channel, seg[:, :, 1], contains object types.
        geom_ids = segmentation[:, :, 0]
        # Infinity is mapped to -1
        geom_ids = geom_ids.astype(np.float64) + 1
        # Scale to [0, 1]
        geom_ids = geom_ids / geom_ids.max()
        pixels = 255*geom_ids
        return pixels.astype(np.uint8)
    
    def get_pointcloud(self, bounds=None, world_coords=True):
        """Returns the point cloud of the scene in either camera or world coordinates."""
        depth = self.get_depth_meters()
        fx, fy = self.I[0, 0], self.I[1, 1]
        cx, cy = self.I[0, 2], self.I[1, 2]
        point_cloud = depth_to_pointcloud(depth, fx, fy, cx, cy, bounds=bounds)
        if world_coords:
            point_cloud.transform(np.linalg.inv(self.E))
        return point_cloud
    
    def get_rgb_pointcloud(self, bounds=None, world_coords=True):
        """Returns the RGB point cloud of the scene in either camera or world coordinates."""
        point_cloud = self.get_pointcloud(bounds=bounds, world_coords=world_coords)
        rgb = self.get_rgb_image()
        colors = rgb.reshape(-1, 3)
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
        return point_cloud
    
    def get_segmented_pointcloud(self, bounds=None, exclude_geom_id=None, world_coords=True, groups=None):
        """Returns the segmented point cloud of the scene in either camera or world coordinates."""
        depth = self.get_depth_meters()
        fx, fy = self.I[0, 0], self.I[1, 1]
        cx, cy = self.I[0, 2], self.I[1, 2]
        seg = self.get_segmentation()
        point_cloud, color_map = depth_to_pointcloud(depth, fx, fy, cx, cy, bounds=bounds, segmentation_image=seg, geom_id_exclude=exclude_geom_id, groups=groups)
        if world_coords:
            point_cloud.transform(np.linalg.inv(self.E))
        return point_cloud, color_map

    def get_camera_axes(self, scale=1.0):
        """
        Generate a visual representation of a camera's local coordinate system using the inverse of its extrinsic parameters.

        Parameters:
        - extrinsics: 4x4 numpy array, representing the camera's extrinsic matrix.
        - scale: float, the length of each axis.

        Returns:
        - Open3D LineSet object representing the camera's axes.
        """
        # Compute the inverse of the extrinsics matrix
        self.update_camera_matrices()
        inv_extrinsics = np.linalg.inv(self.E)
        R = inv_extrinsics[:3, :3]
        t = inv_extrinsics[:3, 3]

        # Define the origin and axes endpoints in the world coordinates
        origin = t
        x_axis = t + R @ np.array([scale, 0, 0])
        y_axis = t + R @ np.array([0, scale, 0])
        z_axis = t + R @ np.array([0, 0, scale])

        # Points and lines for the LineSet
        points = [origin, x_axis, y_axis, z_axis]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue

        # Create the LineSet object
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

if __name__ == "__main__":
    # load point cloud in open3d
    pcd = o3d.geometry.PointCloud()
    #for i in range(10):
    #    pcd += o3d.io.read_point_cloud(f"/home/sammoore/Documents/PointCloud-PointForce/Balanced_Data/legs_pc_{i}.ply")
    for i in range(100):
        pcd += o3d.io.read_point_cloud(f"/home/sammoore/Documents/PointCloud-PointForce/Balanced_Data/body_pc_{i}.ply")
    #n_down_sample = 500
    #idcs = np.random.choice(np.arange(len(pcd.points)), n_down_sample, replace=False)
    #pcd = pcd.select_by_index(idcs)
    pcd0 = o3d.io.read_point_cloud("/home/sammoore/Documents/PointCloud-PointForce/Balanced_Data/legs_pc_123.ply")
    #n_down_sample = 4500
    #idcs = np.random.choice(np.arange(len(pcd0.points)), n_down_sample, replace=False)
    #pcd0 = pcd0.select_by_index(idcs)
    pcd += pcd0
    o3d.visualization.draw_geometries([pcd])
    # Example usage
    #model = mujoco.MjModel.from_xml_path("/home/sammoore/Documents/PointCloud-PointForce/Robots/unitree_go2/go2.xml")
    #data = mujoco.MjData(model)
    #renderer = mujoco.Renderer(model, 480, 640)
    #data.qpos[:] = np.zeros_like(data.qpos)
    #mujoco.mj_step(model, data)

    #camera = Camera("camera", model, data, renderer, lookat=[0, 0, 0], distance=2.0, azimuth=90, elevation=0)
    #pc_seg = camera.get_segmented_pointcloud(bounds=[-2, 2, -2, 2, -2, 2], exclude_geom_id=[0, 1, 2, 3, 4, 5])
    #o3d.visualization.draw_geometries([pc_seg])
    #depth = camera.get_depth_image()
    # get segmentation image
    #segmentation = camera.get_segmentation()
    #geom_types = segmentation[:, :, 1]
    #geom_ids = segmentation[:, :, 0]
    #unique_id = np.unique(geom_ids)
    #unique_type = np.unique(geom_types)
    #obj_ = [mujoco.mjtObj.mjOBJ_UNKNOWN, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_CAMERA, mujoco.mjtObj.mjOBJ_LIGHT, mujoco.mjtObj.mjOBJ_HFIELD, mujoco.mjtObj.mjOBJ_TEXTURE, mujoco.mjtObj.mjOBJ_MATERIAL, mujoco.mjtObj.mjOBJ_MESH, mujoco.mjtObj.mjOBJ_FRAME, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_SENSOR, mujoco.mjtObj.mjOBJ_NUMERIC, mujoco.mjtObj.mjOBJ_TEXT]
    #for i in range(len(unique_id)):
    #    for j in range(len(obj_)):
    #        name = mujoco.mj_id2name(model, obj_[j], unique_id[i])
    #        print(f"{unique_id[i]}: {name}")
    # save segmentation image
    #media.write_image("segmentation.png", segmentation)
    # print unique values in segmentation image
    #print(np.unique(segmentation))
    # cast all zeros to -1
    #geom_ids = geom_ids.astype(np.float64) + 1
    #print(np.unique(geom_ids))
    # mask all zeros
    #geom_ids = np.where(geom_ids == 1, 0, geom_ids)
    #geom_ids = np.where(geom_ids == 2, 0, geom_ids)
    #geom_ids = np.where(geom_ids == 3, 0, geom_ids)
    #geom_ids = np.where(geom_ids == 4, 0, geom_ids)
    #geom_ids = np.where(geom_ids == 5, 0, geom_ids)
    #geom_ids = np.where(geom_ids == 5, 0, geom_ids)

    
    # scale to [0, 255]
    #geom_ids = geom_ids / geom_ids.max()
    #pixels = 255*geom_ids

    #plot.imshow(pixels)
    #plot.savefig("segmentation.png")
    #plot.show()
