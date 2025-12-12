import numpy as np
import torch
import pyvista as pv
import matplotlib.pyplot as plt
import pandas as pd
from pl_module import VSM, BlendedSDF, DeformationSDF, ClassificationVSM, CorrespondenceVSM
import mujoco
from skimage import measure
import plotly.graph_objects as go
import open3d as o3d




def plot_loss(log_dir, term='train_loss'):

    metrics = pd.read_csv(f'{log_dir}/metrics.csv')
    # average the loss over 100 steps
    metrics[term] = metrics[term].rolling(window=10).mean()
    plot = metrics.plot(x='step', y=[term])
    plot.set_xlabel('Step')
    plot.set_ylabel('Loss')
    #plot.set_ylim(9e-3, 9e-2)
    plot.set_yscale('log')
    #plot.set_xlim(0, 100000)
    plot.set_title('Training Loss')
    plot.get_figure().savefig(f'{log_dir}/{term}.png')

def get_sdf_stage_0(log_dir, state_data_path, grid_size=200, bound_box=(-.65, .65), idx=None, nominal=False):
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

    if idx is None:
        idx = np.random.choice(np.arange(0, 100000), 1)
    state = np.load(f"{state_data_path}/joint_data_{idx[0]}.npy")
    if nominal:
        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
        state = state[7:]
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
    return sdf, idx, pose

def get_sdf_stage_1(log_dir, state_data_path, grid_size=200, bound_box=(-.65, .65), idx=None):
    model = BlendedSDF.load_from_checkpoint(f'{log_dir}/checkpoints/newest.ckpt').to('cuda')
    model.eval()
    #model.model.stage1()

    lb = bound_box[0]
    ub = bound_box[1]
    x = np.linspace(lb, ub, num=grid_size)
    y = np.linspace(lb, ub, num=grid_size)
    z = np.linspace(lb, ub, num=grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    if idx is None:
        idx = np.random.choice(np.arange(0, 100000), 1)
    state = np.load(f"{state_data_path}/joint_data_{idx[0]}.npy")
    state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
    state = state[7:]
    mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
    mjdata = mujoco.MjData(mjmodel)
    joint_limits = mjmodel.jnt_range[1:, :]
    state = 2*(state - joint_limits[:, 0])/(joint_limits[:, 1] - joint_limits[:, 0]) - 1
    state = np.repeat(state[np.newaxis, :], points.shape[0], axis=0)
    states = torch.from_numpy(state).float().to('cuda')
    #model = SwitchingSDFModule.load_from_checkpoint(f'/home/sammoore/Documents/PointCloud-PointForce/lightning_logs/version_156/checkpoints/newest.ckpt').to('cuda')
    siren = model.sdf_net
    switch = model.blend_net
    input = torch.cat((torch.from_numpy(points).float().to('cuda'), states), dim=1)

    batch_size = 500  # Reduced batch size
    sdf = []

    with torch.no_grad():
        model.eval()
        #siren.eval()
        #switch.eval()
        for i in range(0, points.shape[0], batch_size):
            input0 = input[i:i+batch_size]
            sdf0 = siren(input0)
            #sdf0 = siren(input0
            switch0 = torch.sigmoid(switch(input0) - 6)
            #switch0 = switch(input0)
            sdf0 = torch.mul((1-switch0), sdf0) + torch.mul(switch0, input0[:, :3].norm(dim=1).unsqueeze(-1))
            sdf.append(sdf0.cpu())

    sdf = torch.cat(sdf)

    sdf = sdf.reshape((grid_size, grid_size, grid_size))
    sdf = sdf.cpu().detach().numpy()
    return sdf, idx
    
def pv_sdf(sdf, level):
    grid_size = sdf.shape[0]
    grid = pv.ImageData()
    grid.dimensions = np.array([grid_size, grid_size, grid_size]) 
    grid.spacing = [2/grid_size]*3
    grid.origin = [0, 0, 0]
    grid.point_data['values'] = sdf.flatten(order='F')

    contours = grid.contour(isosurfaces=[level], method='marching_cubes')
    p = pv.Plotter()
    p.add_mesh(contours, color='white')
    p.show()

def plot_isosurfaces(sdf, iso_levels=[0.0, 0.03, 0.06, 0.09, 0.12, 0.15], opacities=[1, 0.5, 0.25, 0.125, 0.075, 0.0375]):


    if len(opacities) != len(iso_levels):
        raise ValueError("Length of opacities should be equal to the length of iso_levels")

    # Create a 3D plot with Plotly
    fig = go.Figure()

    for level in iso_levels:
        vertices, faces, normals, values = measure.marching_cubes(sdf, level=level)
        
        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=f"Iso-surface level {level}",
            opacity=opacities[iso_levels.index(level)],
        )
        
        fig.add_trace(mesh)

    # Update layout for better visualization
    fig.update_layout(
        title="Multiple Iso-surfaces of SDF",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()

def plot_sdf_slice(sdf, axis, slice_index, levels, version):
    if axis == 'xy':
        sdf_slice = sdf[:, :, slice_index]
        x, y = np.meshgrid(np.arange(sdf.shape[0]), np.arange(sdf.shape[1]))
    elif axis == 'xz':
        sdf_slice = sdf[:, slice_index, :]
        x, y = np.meshgrid(np.arange(sdf.shape[0]), np.arange(sdf.shape[2]))
    elif axis == 'yz':
        sdf_slice = sdf[slice_index, :, :]
        x, y = np.meshgrid(np.arange(sdf.shape[1]), np.arange(sdf.shape[2]))
    
    fig, ax = plt.subplots()
    contour = ax.contourf(x, y, sdf_slice.T, levels=levels, cmap='viridis')
    fig.colorbar(contour)
    ax.set_title(f'SDF contour plot on {axis} plane at slice {slice_index}')
    ax.set_xlabel(axis[0].upper())
    ax.set_ylabel(axis[1].upper())
    plt.savefig(f'./lightning_logs/version_{version}/slice_{axis}_{slice_index}.png')
    plt.show()

def rescale_vertices(vertices, bound_box, grid_size):
    # Calculate the physical length of each voxel along each axis
    voxel_size = (bound_box[1] - bound_box[0]) / grid_size
    # Rescale vertices from grid indices to physical coordinates
    vertices = vertices * voxel_size + bound_box[0]
    return vertices

def plot_correspondences(sdf_log_dir, correspondences_log_dir):
    corr_net_base = CorrespondenceVSM.load_from_checkpoint(f"{correspondences_log_dir}/checkpoints/newest.ckpt").to('cuda')
    corr_net = corr_net_base.model

    corr_net_inv = corr_net_base.model_inv

    state_data_path = './Balanced_Data'
    sdf1, idx1, pose1 = get_sdf_stage_0(sdf_log_dir, state_data_path, grid_size=300)
    sdf2, idx2, pose2 = get_sdf_stage_0(sdf_log_dir, state_data_path, grid_size=300)
    sdf3, idx3, pose3 = get_sdf_stage_0(sdf_log_dir, state_data_path, grid_size=300, nominal=True)
    # solve for zero level sets
    print("solving for zero level sets")
    vertices1, faces1, normals1, values1 = measure.marching_cubes(sdf1, level=0.0)
    vertices2, faces2, normals2, values2 = measure.marching_cubes(sdf2, level=0.0)
    vertices3, faces3, normals3, values3 = measure.marching_cubes(sdf3, level=0.0)

    # convert to original scale
    vertices1 = rescale_vertices(vertices1, (-.65, .65), 300)
    vertices2 = rescale_vertices(vertices2, (-.65, .65), 300)
    vertices3 = rescale_vertices(vertices3, (-.65, .65), 300)

    # get the corresponding point on the second mesh
    # with torch.no_grad():
    #     deform_net.eval()
    #     input = torch.cat((point, torch.from_numpy(pose2).float().to('cuda').unsqueeze(0)), dim=1)
    #     verts = torch.from_numpy(vertices2).float().to('cuda')
    #     inputs = torch.cat((verts, torch.from_numpy(pose2).float().to('cuda').unsqueeze(0).repeat(verts.shape[0], 1)), dim=1)
    #     delta_p = deform_net(input)
    #     point2 = point + delta_p[:, :3]
    #     new_verts = verts + deform_net(inputs)[:,:3]
    #     new_verts = new_verts.cpu().numpy()
    
    with torch.no_grad():
        corr_net.eval()
        input1 = torch.cat((torch.from_numpy(vertices1).float().to('cuda'), torch.from_numpy(pose1).float().to('cuda').unsqueeze(0).repeat(vertices1.shape[0], 1)), dim=1)
        input2 = torch.cat((torch.from_numpy(vertices2).float().to('cuda'), torch.from_numpy(pose2).float().to('cuda').unsqueeze(0).repeat(vertices2.shape[0], 1)), dim=1)

        feat1 = corr_net(input1)
        feat2 = corr_net(input2)

        feat1 = feat1.cpu().numpy()
        feat2 = feat2.cpu().numpy()

        feat1 += vertices1
        feat2 += vertices2

        input3 = torch.cat((torch.from_numpy(vertices3).float().to('cuda'), torch.from_numpy(pose2).float().to('cuda').unsqueeze(0).repeat(vertices3.shape[0], 1)), dim=1)
        
        feat3 = corr_net_inv(input3)
        feat3 = feat3.cpu().numpy()
        feat3 += vertices3



    # plot as mesh
    mesh1 = o3d.geometry.TriangleMesh()
    mesh1.vertices = o3d.utility.Vector3dVector(vertices1)
    mesh1.triangles = o3d.utility.Vector3iVector(faces1)
    mesh1.compute_vertex_normals()
    colors = (feat1 - np.min(vertices3, axis=0)) / (np.max(vertices3, axis=0) - np.min(vertices3, axis=0))
    mesh1.vertex_colors = o3d.utility.Vector3dVector(colors)

    # shift the second mesh to the right
    vertices2[:, 1] += .8
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(vertices2)
    mesh2.triangles = o3d.utility.Vector3iVector(faces2)
    mesh2.compute_vertex_normals()
    colors = (feat2 - np.min(vertices3, axis=0)) / (np.max(vertices3, axis=0) - np.min(vertices3, axis=0))
    mesh2.vertex_colors = o3d.utility.Vector3dVector(colors)

    vertices3[:, 1] -= .8
    mesh3 = o3d.geometry.TriangleMesh()
    mesh3.vertices = o3d.utility.Vector3dVector(vertices3)
    mesh3.triangles = o3d.utility.Vector3iVector(faces3)
    mesh3.compute_vertex_normals()
    colors = (vertices3 - np.min(vertices3, axis=0)) / (np.max(vertices3, axis=0) - np.min(vertices3, axis=0))
    mesh3.vertex_colors = o3d.utility.Vector3dVector(colors)

    vertices3[:, 1] += .8
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(feat3)
    colors = (vertices3 - np.min(vertices3, axis=0)) / (np.max(vertices3, axis=0) - np.min(vertices3, axis=0))
    pc.colors = o3d.utility.Vector3dVector(colors)

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(vertices2)
    colors = (feat2 - np.min(vertices3, axis=0)) / (np.max(vertices3, axis=0) - np.min(vertices3, axis=0))
    pc2.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([mesh1, mesh2, mesh3])
    o3d.visualization.draw_geometries([pc, pc2])
    
def plot_classifier(sdf_log_dir, classifier_log_dir):
    classifier = ClassificationVSM.load_from_checkpoint(classifier_log_dir).to('cuda')
    classifier = classifier.model

    state_data_path = './Balanced_Data'
    sdf, idx, pose = get_sdf_stage_0(sdf_log_dir, state_data_path, nominal=True)
    vertices, faces, normals, values = measure.marching_cubes(sdf, level=0.0)
    vertices = rescale_vertices(vertices, (-.65, .65), 200)
    vertices = torch.from_numpy(vertices).float().to('cuda')
    #body_pc_path = "/home/sammoore/Documents/PointCloud-PointForce/classification_data/rigid_link_pc_0_BL_Hip.ply"
    #body_pc = o3d.io.read_point_cloud(body_pc_path)
    #pose = torch.from_numpy(pose).float().to('cuda')
    #inputs = torch.cat((vertices, pose.unsqueeze(0).repeat(vertices.shape[0], 1)), dim=1)
    with torch.no_grad():
        classifier.eval()
        outputs = classifier(vertices)
        outputs = torch.argmax(outputs, dim=1)
        outputs = outputs.cpu().numpy()

    # outputs to colors
    classes = np.unique(outputs)
    print(classes)    
    colors = np.zeros((outputs.shape[0], 3))
    for i, c in enumerate(classes):
        colors[outputs == c] = np.random.rand(3)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])




if __name__ == '__main__':
    version = 176
    state_data_path = './Balanced_Data'
    #plot_loss(log_dir="/home/sammoore/Documents/PointCloud-PointForce/stage_2_logs/trial_0", term='train_loss')
    #plot_loss(log_dir="/home/sammoore/Documents/PointCloud-PointForce/stage_2_logs/trial_0", term='dist_loss')
    plot_loss(log_dir="/home/sammoore/Documents/PointCloud-PointForce/correspondence_logs/trial_0", term='train_loss')
    #plot_loss(log_dir="/home/sammoore/Documents/PointCloud-PointForce/classification_logs/trial_0", term='train_loss')
    #plot_loss(log_dir="/home/sammoore/Documents/PointCloud-PointForce/stage_2_logs/trial_0", term='normal_loss')
    #sdf, idx = get_sdf_stage_0('/home/sammoore/Downloads/split_siren/', state_data_path, grid_size=250, bound_box=(-.65, .65))
    #pv_sdf(sdf)
    #sdf, idx = get_sdf_stage_0('/home/sammoore/Documents/PointCloud-PointForce/stage_0_architecture_logs/mod_siren_split_trial_0/', state_data_path, grid_size=250, bound_box=(-.65, .65))
    #pv_sdf(sdf)
    #sdf, idx = get_sdf_stage_1('/home/sammoore/Documents/PointCloud-PointForce/stage_1_logs/siren_split_trial_0', state_data_path, grid_size=400, bound_box=(-1, 1))
    #print("got sdf")
    #pv_sdf(sdf, level=0.0)
    #plot_isosurfaces(sdf, iso_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], opacities=[1, 0.5, 0.25, 0.125, 0.075, 0.0375])
    
    # Plot SDF slices on different planes
    #plot_sdf_slice(sdf, 'xy', slice_index=80, levels=20, version=version)
    #plot_sdf_slice(sdf, 'xz', slice_index=80, levels=20, version=version)
    #plot_sdf_slice(sdf, 'yz', slice_index=80, levels=20, version=version)

    plot_correspondences(sdf_log_dir="/home/sammoore/Downloads/split_siren/", correspondences_log_dir="/home/sammoore/Documents/PointCloud-PointForce/correspondence_logs/trial_0")
    #plot_classifier(sdf_log_dir="/home/sammoore/Downloads/split_siren/", classifier_log_dir="/home/sammoore/Documents/PointCloud-PointForce/classification_logs/trial_0/checkpoints/newest.ckpt")
