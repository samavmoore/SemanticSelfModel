import torch
import os
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
import mujoco
import time


class VSMDataset(Dataset):
    def __init__(self, data_dir, mode='train_on_surf'):

        modes = ['train_on_surf', 'test_on_surf', 'train_off_surf', 'test_off_surf', 'curriculum']
        if mode not in modes:
            raise ValueError(f"Invalid mode '{mode}' specified. Must be one of {modes}")
        self.mode = mode
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.startswith('joint_data')] # just to get the number of files
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0
        self.z_min, self.z_max = 0, 0
        mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
        joint_limits = mjmodel.jnt_range[1:, :]
        self.joint_mins = joint_limits[:, 0]
        self.joint_maxs = joint_limits[:, 1]
        
        if mode == 'train_on_surf':
            self.start_idx = 0
            self.stop_idx = 83200
        elif mode == 'test_on_surf':
            self.start_idx = 83200
            self.stop_idx = 102400
        elif mode == 'train_off_surf':
            self.start_idx = 0
            self.stop_idx = 83200
        elif mode == 'test_off_surf':
            self.start_idx = 83200
            self.stop_idx = 102400
        elif mode == 'curriculum':
            self.start_idx = 0
            self.stop_idx = 7500

    def __len__(self):
        return self.stop_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx
        joint_data_file = os.path.join(self.data_dir, f'joint_data_{idx}.npy')
        
        legs_pc_file = os.path.join(self.data_dir, f'legs_pc_{idx}.ply')
        off_pc_file = os.path.join(self.data_dir, f'off_surface_pc_{idx}.ply')
        off_d_file = os.path.join(self.data_dir, f'off_surface_d_{idx}.npy')

        body_idx = np.random.randint(0, 7000) # stopped collecting body point clouds after 7000
        body_pc_file = os.path.join(self.data_dir, f'body_pc_{body_idx}.ply')

        joint_data = np.load(joint_data_file)
        body_pc = o3d.io.read_point_cloud(body_pc_file)
        legs_pc = o3d.io.read_point_cloud(legs_pc_file)
        off_pc = o3d.io.read_point_cloud(off_pc_file)
        off_d = np.load(off_d_file)

        body_pc_points = np.asarray(body_pc.points)
        body_pc_normals = np.asarray(body_pc.normals)
        body_pc_sdf = np.zeros_like(body_pc_points[:, 0])

        legs_pc_points = np.asarray(legs_pc.points)
        legs_pc_normals = np.asarray(legs_pc.normals)
        legs_pc_sdf = np.zeros_like(legs_pc_points[:, 0])


        off_pc_points = np.asarray(off_pc.points)
        off_pc_normals = np.asarray(off_pc.normals)
        off_pc_sdf = off_d

        # normalize the joint data
        if np.any(self.joint_maxs == self.joint_mins):
            joint_data = np.zeros_like(joint_data)
            joint_data_scaled = 2 * (joint_data - self.joint_mins) / (self.joint_maxs - self.joint_mins) - 1
            joint_data = np.where(self.joint_maxs == self.joint_mins, joint_data, joint_data_scaled)
        else:
            joint_data = 2 * (joint_data - self.joint_mins) / (self.joint_maxs - self.joint_mins) - 1
        
        
        # calculate the min and max values for the body and legs point clouds
        self.x_min = min(np.stack((self.x_min, np.min(legs_pc_points[:, 0])), axis=0))
        self.x_max = max(np.stack((self.x_max, np.max(legs_pc_points[:, 0])), axis=0))
        self.y_min = min(np.stack((self.y_min, np.min(legs_pc_points[:, 1])), axis=0))
        self.y_max = max(np.stack((self.y_max, np.max(legs_pc_points[:, 1])), axis=0))
        self.z_min = min(np.stack((self.z_min, np.min(legs_pc_points[:, 2])), axis=0))
        self.z_max = max(np.stack((self.z_max, np.max(legs_pc_points[:, 2])), axis=0))

        joint_data = np.repeat(joint_data[np.newaxis, :], 500, axis=0)

        # downsample the point clouds
        idx_body = np.random.choice(np.arange(len(body_pc_points)), 100)
        idx_legs = np.random.choice(np.arange(len(legs_pc_points)), 400)
        idx_off_pc = np.random.choice(np.arange(len(off_pc_points)), 500)


        off_pc_points = off_pc_points[idx_off_pc]
        off_pc_sdf = off_pc_sdf[idx_off_pc]
        off_pc_normals = off_pc_normals[idx_off_pc]

        if self.mode in ['train_on_surf', 'test_on_surf', 'curriculum']:
            off_pc_points_x = np.random.uniform(self.x_min, self.x_max, (500, 1))
            off_pc_points_y = np.random.uniform(self.y_min, self.y_max, (500, 1))
            off_pc_points_z = np.random.uniform(self.z_min, self.z_max, (500, 1))
            off_pc_points = np.concatenate((off_pc_points_x, off_pc_points_y, off_pc_points_z), axis=1)


        body_pc_points = body_pc_points[idx_body]
        body_pc_normals = body_pc_normals[idx_body]
        body_pc_sdf = body_pc_sdf[idx_body]

        legs_pc_points = legs_pc_points[idx_legs]
        legs_pc_normals = legs_pc_normals[idx_legs]
        legs_pc_sdf = legs_pc_sdf[idx_legs]

        on_pc_points = np.concatenate((body_pc_points, legs_pc_points), axis=0)
        on_pc_normals = np.concatenate((body_pc_normals, legs_pc_normals), axis=0)
        on_pc_sdf = np.concatenate((body_pc_sdf, legs_pc_sdf), axis=0)

        return {
            'joint_data': torch.from_numpy(joint_data).float(),
            'on_pc_points': torch.from_numpy(on_pc_points).float(),
            'on_pc_normals': torch.from_numpy(on_pc_normals).float(),
            'on_pc_sdf': torch.from_numpy(on_pc_sdf).float(),
            'off_pc_points': torch.from_numpy(off_pc_points).float(),
            'off_pc_normals': torch.from_numpy(off_pc_normals).float(),
            'off_pc_sdf': torch.from_numpy(off_pc_sdf).float(),
        }
    



class VSMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=1, num_workers=4, split_ratio=(0.8, 0.1, 0.1), mode='train'):
        super().__init__()
        """
        Args:
            data_dir: Path to the directory containing the data
            batch_size: Batch size for the dataloaders
            num_workers: Number of workers for the dataloaders
            split_ratio: Tuple containing the ratios for the train, validation, and test sets
            mode: Specifies the mode of the dataset ('train_on_surf', 'test_on_surf', 'train_off_surf', 'test_off_surf', 'curriculum')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mode = mode

    def prepare_data(self):
        # Load full dataset (without indices)
        self.full_dataset = VSMDataset(self.data_dir, self.mode)
        total_size = len(self.full_dataset)
        train_end = int(total_size * self.split_ratio[0])
        val_end = train_end + int(total_size * self.split_ratio[1])

        # Shuffle indices
        indices = np.random.permutation(total_size)
        self.train_idx = indices[:train_end]
        self.val_idx = indices[train_end:val_end]
        self.test_idx = indices[val_end:]

    def setup(self, stage=None):
        # Assign Subsets based on the indices computed in prepare_data
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.full_dataset, self.train_idx)
            self.val_dataset = Subset(self.full_dataset, self.val_idx)
        if stage == 'test' or stage is None:
            self.test_dataset = Subset(self.full_dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

def convert_txt_to_npy(directory, prefix):
    # List all files in the directory with the specified prefix
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.npy')]

    for file in files:
        file_path = os.path.join(directory, file)
        # Load the content as text
        data = np.loadtxt(file_path)

        # Save the array back to the same name, effectively replacing it
        np.save(file_path, data)

        print(f"Processed '{file}'")



class SemanticVSMDataset(Dataset):
    def __init__(self, data_dir, mode='train'):

        modes = ['train', 'test']
        if mode not in modes:
            raise ValueError(f"Invalid mode '{mode}' specified. Must be one of {modes}")
        self.mode = mode
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.startswith('joint_data')] # just to get the number of files
        self.x_min, self.x_max = 0, 0
        self.y_min, self.y_max = 0, 0
        self.z_min, self.z_max = 0, 0
        mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
        joint_limits = mjmodel.jnt_range[1:, :]
        self.joint_mins = joint_limits[:, 0]
        self.joint_maxs = joint_limits[:, 1]
    
        
        if mode == 'train':
            self.start_idx = 0
            self.stop_idx = 19999
        elif mode == 'test':
            self.start_idx = 16000
            self.stop_idx = 20000

    def __len__(self):
        return self.stop_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx
        joint_data_file = os.path.join(self.data_dir, f'joint_data_{idx}.npy')
        joint_data = np.load(joint_data_file)

        pc, colors = self.load_point_cloud(f'pc_{idx}.ply', n_points=5000)

        # normalize the joint data
        if np.any(self.joint_maxs == self.joint_mins):
            joint_data = np.zeros_like(joint_data)
            joint_data_scaled = 2 * (joint_data - self.joint_mins) / (self.joint_maxs - self.joint_mins) - 1
            joint_data = np.where(self.joint_maxs == self.joint_mins, joint_data, joint_data_scaled)
        else:
            joint_data_normalized = 2 * (joint_data - self.joint_mins) / (self.joint_maxs - self.joint_mins) - 1
        
        joint_data = np.repeat(joint_data[np.newaxis, :], 5000, axis=0)
        joint_data_normalized = np.repeat(joint_data_normalized[np.newaxis, :], 5000, axis=0)


        return {'joint_data_rad': torch.from_numpy(joint_data).float(),
                'joint_data_scaled': torch.from_numpy(joint_data_normalized).float(),
                'xyz': torch.from_numpy(pc).float(),
                'colors': torch.from_numpy(colors).float(),
        }

    def load_point_cloud(self, pc_file, n_points=None):
        pc_file = os.path.join(self.data_dir, pc_file)
        pc = o3d.io.read_point_cloud(pc_file)
        pc_points = np.asarray(pc.points)
        pc_colors = np.asarray(pc.colors)
        if n_points is not None:
            idx = np.random.choice(np.arange(len(pc_points)), n_points)
            pc_points = pc_points[idx]
            pc_colors = pc_colors[idx]
        return pc_points, pc_colors



class SemanticVSMDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=1, num_workers=4, split_ratio=(1.0, 0., 0.), mode='train'):
        super().__init__()
        """
        Args:
            data_dir: Path to the directory containing the data
            batch_size: Batch size for the dataloaders
            num_workers: Number of workers for the dataloaders
            split_ratio: Tuple containing the ratios for the train, validation, and test sets
            mode: Specifies the mode of the dataset ('train', 'test')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratio = split_ratio
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.mode = mode

    def prepare_data(self):
        # Load full dataset (without indices)
        self.full_dataset = SemanticVSMDataset(self.data_dir, self.mode)
        total_size = len(self.full_dataset)
        train_end = int(total_size * self.split_ratio[0])
        val_end = train_end + int(total_size * self.split_ratio[1])

        # Shuffle indices
        indices = np.random.permutation(total_size)
        self.train_idx = indices[:train_end]
        self.val_idx = indices[train_end:val_end]
        self.test_idx = indices[val_end:]

    def setup(self, stage=None):
        # Assign Subsets based on the indices computed in prepare_data
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.full_dataset, self.train_idx)
            self.val_dataset = Subset(self.full_dataset, self.val_idx)
        if stage == 'test' or stage is None:
            self.test_dataset = Subset(self.full_dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    data_dir = './Debug_Data'

    dm = VSMDataModule(data_dir, batch_size=1, num_workers=4, split_ratio=(0.5, 0.3, 0.2))
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    for batch in train_loader:
        print(batch['joint_data'].shape)
        print(batch['on_pc_points'].shape)
        print(batch['on_pc_normals'].shape)
        print(batch['on_pc_sdf'].shape)
        print(batch['off_pc_points'].shape)
        print(batch['off_pc_normals'].shape)
        print(batch['off_pc_sdf'].shape)
        break