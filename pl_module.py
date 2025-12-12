import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models import StateConditionMLPQueryModel, QuadrupedSDFSiren, QuadrupedSDFRelu, StateConditionMLPQueryModelSplit, StateConditionMLPQueryModelSplitFuse, QuadrupedSDFSirenSplit, SirenLayer, TransformationSiren
from torch.func import grad, vmap, jacrev
from pl_datamodule import VSMDataModule
import numpy as np
import pyvista as pv
import pandas as pd
import os
import mujoco
import json


def frange_cycle_linear(n_iters, pretrain_period, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    if start-stop == 0:
        return np.ones(n_iters + pretrain_period) * stop
    
    L = np.ones(n_iters + pretrain_period) * stop
    L[:pretrain_period] = 0  # Setting the pretraining period coefficients to 0
    
    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period + pretrain_period) < n_iters + pretrain_period):
            L[int(i + c * period + pretrain_period)] = v
            v += step
            i += 1
            
    return L 

def frange_cycle_sigmoid(n_iters, pretrain_period, start = 0.0, stop = 1.0,  n_cycle=4, ratio=0.5):
    if start-stop == 0:
        return np.ones(n_iters + pretrain_period) * stop
    
    start0 = start
    start = 0.0
    stop0 = stop
    stop = 1.0
    
    L = np.ones(n_iters+pretrain_period)*stop
    L[:pretrain_period] = 0 # Setting the pretraining period coefficients to 0

    period = n_iters/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop:
            idx = int(i+c*period) + pretrain_period
            L[idx] = 1.0/(1.0 + np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L*(stop0-start0) + start0


class CyclicalAnnealingCallback(pl.Callback):
    def __init__(self, 
                 max_epochs, 
                 steps_per_epoch, 
                 pretrain_epochs, 
                 start=(0., 0., 0.), 
                 stop=(1, 1., 1.), 
                 n_cycle=(1, 3, 4), 
                 ratio=(.5, .6, .8), 
                 annealing=True,
                 reg_annealing=False, 
                 type='linear'):
        """
        Implements cyclical annealing for the loss weights and discount factor

        Args:

        max_epochs: maximum number of epochs to train for
        steps_per_epoch: number of steps per epoch
        pretrain_epochs: number of epochs to pretrain for
        start: start values for the schedule, tuple of length 5 (chain rule, prediction, state prediction, gamma, G_reg)
        stop: stop values for the schedule, tuple of length 5
        n_cycle: number of cycles for each parameter, tuple of length 5
        ratio: ratio of the linear schedule to the constant schedule, tuple of length 5
        annealing: whether or not to anneal the loss weights and discount factor
        reg_annealing: whether or not to anneal the regularization weights
        type: type of annealing, either 'linear' or 'sigmoid'
        """


        self.annealing = annealing
        self.reg_annealing = reg_annealing
        n_cycle1, n_cycle2, n_cycle3 = n_cycle
        start1, start2, start3= start
        stop1, stop2, stop3= stop
        ratio1, ratio2, ratio3 = ratio
        n_iter = max_epochs * steps_per_epoch
        self.pretrain_period = pretrain_epochs * steps_per_epoch

        if type == 'linear':
            self.schedule_1 = frange_cycle_linear(n_iter, self.pretrain_period, start1, stop1, n_cycle1, ratio1)
            self.schedule_2 = frange_cycle_linear(n_iter, self.pretrain_period, start2, stop2, n_cycle2, ratio2)
            self.schedule_3 = frange_cycle_linear(n_iter, self.pretrain_period, start3, stop3, n_cycle3, ratio3)
        elif type == 'sigmoid':
            self.schedule_1 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start1, stop1, n_cycle1, ratio1)
            self.schedule_2 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start2, stop2, n_cycle2, ratio2)
            self.schedule_3 = frange_cycle_sigmoid(n_iter, self.pretrain_period, start3, stop3, n_cycle3, ratio3)
        else:
            raise ValueError("type must be either 'linear' or 'sigmoid'")

        self.init_normals_wt = None
        self.init_grad_wt = None
        self.init_exp_penalty_wt = None

    def on_train_start(self, trainer, pl_module):
        self.init_grad_wt = pl_module.grad_wt
        self.init_exp_penalty_wt = pl_module.exp_penalty_wt
        self.init_normals_wt = pl_module.normals_wt


    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.sdf_level_set_wt = self.schedule_1[trainer.global_step]
        #pl_module.exp_penalty_wt = self.schedule_2[trainer.global_step]
        #pl_module.normals_wt = self.schedule_3[trainer.global_step]


class VSM(pl.LightningModule):
    def __init__(self, architecture='vsm_og',
                  n_states=12, lr=.95e-4, exp_penalty_a=1e2, 
                 grad_a=5e1, normal_constraint_a=1e2, sdf_a=3e3, 
                 exp_scale=1e2, epochs=1000, n_obs=10000, devices=1, batch_size=1):
        super(VSM, self).__init__()
        self.save_hyperparameters()
        architectures = ['siren_og', 'siren_split', 'siren_split_fuse', 'mod_siren', 'mod_relu', 'mod_siren_split']
        assert architecture in architectures, f"Architecture must be one of {architectures}"
        if architecture == 'siren_og':
            self.model = StateConditionMLPQueryModel(in_channels=(3+n_states), out_channels=1, hidden_features=256)
        elif architecture == 'siren_split':
            self.model = StateConditionMLPQueryModelSplit(n_states=12, hidden_features=256)
        elif architecture == 'siren_split_fuse':
            self.model = StateConditionMLPQueryModelSplitFuse(n_states=12, hidden_features=256)
        elif architecture == 'mod_siren':
            self.model = QuadrupedSDFSiren(joint_input_dim=3, xyz_input_dim=3, hidden_dim=136, sdf_output_dim=1, latent_dim=136)
        elif architecture == 'mod_relu':
            self.model = QuadrupedSDFRelu(joint_input_dim=3, xyz_input_dim=3, hidden_dim=136, sdf_output_dim=1, latent_dim=136)
        elif architecture == 'mod_siren_split':
            self.model = QuadrupedSDFSirenSplit(joint_input_dim=3, xyz_input_dim=3, hidden_dim=176, sdf_output_dim=1, latent_dim=176)

        self.architecture = architecture
        self.lr = lr

        self.grad_wt = 1
        self.exp_penalty_wt = 1
        self.normals_wt = 1
        self.sdf_level_set_wt = 0
        self.exp_penalty_a = exp_penalty_a
        self.grad_a = grad_a
        self.normal_constraint_a = normal_constraint_a
        self.sdf_a = sdf_a
        self.exp_scale = exp_scale
        self.epochs = epochs
        self.devices = devices
        self.n_obs = n_obs
        self.batch_size = batch_size

    def forward(self, x):
        on_xyz = x['on_pc_points']
        on_normals = x['on_pc_normals']
        on_sdf = x['on_pc_sdf']
        off_xyz = x['off_pc_points']
        off_normals = x['off_pc_normals']
        off_sdf = x['off_pc_sdf']
        state = x['joint_data']
        
        # new "batch size" is 2000*batch_size
        on_xyz = torch.reshape(on_xyz, (-1, 3))
        on_normals = torch.reshape(on_normals, (-1, 3))
        on_sdf = torch.reshape(on_sdf, (-1, 1))
        off_xyz = torch.reshape(off_xyz, (-1, 3))
        off_normals = torch.reshape(off_normals, (-1, 3))
        off_sdf = torch.reshape(off_sdf, (-1, 1))
        state = torch.reshape(state, (-1, state.shape[2]))

        #off_xyz = torch.rand_like(off_xyz)
        # scale from -1 to 1
        #off_xyz = off_xyz*2 - 1
        xyzs = torch.cat((on_xyz, off_xyz)).requires_grad_(True)
        grads = torch.cat((on_normals, off_normals))
        sdfs = torch.cat((on_sdf, off_sdf))
        states = torch.cat((state, state))

        input = torch.cat((xyzs, states), dim=1)
        
        sdfs_hat = self.model(input)
        #grads_hat0 = vmap(grad(model), in_dims=(0, 0))(xyzs, states)
        grads_hat = torch.autograd.grad(sdfs_hat, xyzs, grad_outputs=torch.ones_like(sdfs_hat), create_graph=True)[0]


        sdf_constraint0 = torch.where(sdfs == 0, torch.abs(sdfs_hat), torch.zeros_like(sdfs_hat)).mean()*self.sdf_a
        sdf_constraint1 =  0 # self.weighted_sdf_loss(sdfs_hat, sdfs, alpha=self.sdf_level_set_wt)*self.sdf_a
        grad_constraint = torch.abs(grads_hat.norm(dim=-1) - 1).mean()*self.grad_a
        eps = 1e-10
        penality_mask = (sdfs_hat < eps) & (sdfs > 0)
        off_body_constraint = torch.where(penality_mask, torch.ones_like(sdfs), torch.zeros_like(sdfs)).mean()
        exp_penalty =  torch.where(sdfs==0, torch.zeros_like(sdfs_hat), torch.exp(-self.exp_scale*torch.abs(sdfs_hat))).mean()*self.exp_penalty_a
        normal_constraint0  = torch.where(sdfs==0, 1-F.cosine_similarity(grads_hat, grads, dim=-1)[..., None], torch.zeros_like(sdfs_hat[..., :1])).mean()*self.normal_constraint_a
        normal_constraint1 =  0 #self.weighted_normal_loss(grads_hat, grads, sdfs, alpha=self.sdf_level_set_wt)*self.normal_constraint_a
        
        loss = (sdf_constraint0 + sdf_constraint1) + grad_constraint + exp_penalty*self.exp_penalty_wt + (normal_constraint0 + normal_constraint1)  #+ off_body_constraint*1e2
        return loss, (sdf_constraint0 + sdf_constraint1)*(1/self.sdf_a), grad_constraint*(1/self.grad_a), exp_penalty*(1/self.exp_penalty_a), (normal_constraint0 + normal_constraint1)*(1/self.normal_constraint_a), off_body_constraint

    def configure_optimizers(self):
        if self.architecture in ['siren_og', 'siren_split', 'siren_split_fuse']:
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.architecture in ['mod_siren', 'mod_relu', 'mod_siren_split']:
            optim = torch.optim.Adam([
            {"params": self.model.kinematic_net.parameters(), "lr": self.lr*.125},
            {"params": self.model.modulator.parameters(), "lr": self.lr*.125},
            {"params": self.model.siren_sdf_net.parameters(), "lr": self.lr}])
        #optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=int(np.ceil(self.n_obs*(1/self.batch_size)*(1/self.devices))))
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, max_lr=self.lr, base_lr=self.lr*.2e-1, step_size_up=int(self.n_obs*(1/5)*(1/self.devices))*4, mode='exp_range', gamma=.99)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=350000, gamma=.5)
        return [optim]  #, [scheduler]

    def training_step(self, batch, batch_idx):
        loss, sdf_constraint, grad_constraint, exp_penalty, normal_constraint, off_body_constraint = self.forward(batch)
        self.log('train_loss', loss)
        self.log('sdf_constraint', sdf_constraint)
        self.log('grad_constraint', grad_constraint)
        self.log('exp_penalty', exp_penalty)
        self.log('normal_constraint', normal_constraint)
        self.log('off_body_constraint', off_body_constraint)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self.forward(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self.forward(batch)
        self.log('test_loss', loss)
        return loss


    def weighted_sdf_loss(self, output, target, alpha=.3):
        # Calculate weights based on target SDF values
        alpha_new = .5e2 + 1 - .5e2*alpha
        weights = torch.exp(-alpha_new * torch.abs(target))
        
        # Calculate absolute differences
        loss = torch.abs(output - target)
        
        # Apply weights
        weighted_loss = weights * loss
        
        # Return mean weighted loss
        return weighted_loss.mean()
    
    def weighted_normal_loss(self, output, target, sdf, alpha=.3):
        # Calculate weights based on target SDF values
        alpha_new = .5e2 + 1 - .5e2*alpha
        weights = torch.exp(-alpha_new * torch.abs(sdf))
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(output, target, dim=-1)
        
        # Calculate absolute differences
        loss = 1 - cos_sim
        
        # Apply weights
        weighted_loss = weights * loss
        
        # Return mean weighted loss
        return weighted_loss.mean()
    
import torch.nn as nn

class BlendedSDF(pl.LightningModule):
    def __init__(self, lr_sdf=0.00001, lr_blend_net=0.0001, pretrained_sdf_path=None, pretrained_blend_net_path=None):
        super(BlendedSDF, self).__init__()
        self.save_hyperparameters()

        pretrained_sdf = VSM.load_from_checkpoint(pretrained_sdf_path).model
        pretrained_blend_net = VSM.load_from_checkpoint(pretrained_blend_net_path).model
        
        self.sdf_net = pretrained_sdf
        self.blend_net = pretrained_blend_net

        
        # Learning rates
        self.lr_sdf = lr_sdf
        self.lr_blend_net = lr_blend_net

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute the old SDF values using the pretrained model
        on_xyz = x['on_pc_points']
        on_normals = x['on_pc_normals']
        on_sdf = x['on_pc_sdf']
        off_xyz = x['off_pc_points']
        off_normals = x['off_pc_normals']
        off_sdf = x['off_pc_sdf']
        state = x['joint_data']
        
        # new "batch size" is 2000*batch_size
        on_xyz = torch.reshape(on_xyz, (-1, 3))
        on_normals = torch.reshape(on_normals, (-1, 3))
        on_sdf = torch.reshape(on_sdf, (-1, 1))
        off_xyz = torch.reshape(off_xyz, (-1, 3))
        off_normals = torch.reshape(off_normals, (-1, 3))
        off_sdf = torch.reshape(off_sdf, (-1, 1))
        state = torch.reshape(state, (-1, state.shape[2]))

        #off_xyz = torch.rand_like(off_xyz)
        # scale from -1 to 1
        #off_xyz = off_xyz*2 - 1
        xyzs = torch.cat((on_xyz, off_xyz)).requires_grad_(True)
        grads = torch.cat((on_normals, off_normals))
        sdfs = torch.cat((on_sdf, off_sdf))
        states = torch.cat((state, state))

        input = torch.cat((xyzs, states), dim=1)

        # Compute the new SDF values using the pretrained model
        sdf_pred_old = self.sdf_net(input)
        switch = self.blend_net(input)
        #alpha = self.alpha(input)
        #rho = self.rho(input)
        norm_xyz = torch.norm(xyzs, dim=1)
        #switch = self.sigmoid(alpha * (norm_xyz.unsqueeze(-1) - rho))
        switch = self.sigmoid(switch - 6)
        blended_sdf = (1 - switch) * sdf_pred_old +  switch* norm_xyz[..., None]
        grads_pred = torch.autograd.grad(blended_sdf, xyzs, grad_outputs=torch.ones_like(blended_sdf), create_graph=True)[0]

        
        return blended_sdf, grads_pred, sdfs, grads

    def compute_losses(self, sdf_pred, grads_pred, sdf_target, normals_target):
        sdf_loss = F.l1_loss(sdf_pred, sdf_target)
        sdf_loss_on_suface = torch.where(sdf_target == 0, torch.abs(sdf_pred), torch.zeros_like(sdf_pred)).mean()
        cos_sim = F.cosine_similarity(grads_pred, normals_target, dim=1)
        cos_sim_on_surface = torch.where(sdf_target == 0, cos_sim, torch.zeros_like(cos_sim))
        normal_loss = 1 - cos_sim.mean()  # 1 - mean cosine similarity
        #normal_loss = self.weighted_normal_loss(grads_pred, normals_target, sdf_target)
        #normal_loss_on_surface = 1 - cos_sim_on_surface.mean()
        grad_norm_loss = torch.abs(grads_pred.norm(dim=1) - 1).mean()
        total_loss = sdf_loss + normal_loss + grad_norm_loss #+ 10*sdf_loss_on_suface + 10*normal_loss_on_surface
        return total_loss, sdf_loss, normal_loss, grad_norm_loss, sdf_loss_on_suface#, normal_loss_on_surface

    def configure_optimizers(self):
        # Separate parameter groups with different learning rates
        #optim = torch.optim.Adam([
        #{"params": self.sdf_net.kinematic_net.parameters(), "lr": self.lr_sdf*.1},
        #{"params": self.sdf_net.modulator.parameters(), "lr": self.lr_sdf*.1},
        #{"params": self.sdf_net.siren_sdf_net.parameters(), "lr": self.lr_sdf}, 
        #{"params": self.blend_net.kinematic_net.parameters(), "lr": self.lr_blend_net*.1},
        #{"params": self.blend_net.modulator.parameters(), "lr": self.lr_blend_net*.1},
        #{"params": self.blend_net.siren_sdf_net.parameters(), "lr": self.lr_blend_net} ])
        optim = torch.optim.Adam([
        {"params": self.sdf_net.parameters(), "lr": self.lr_sdf},
        {"params": self.blend_net.parameters(), "lr": self.lr_blend_net}])
        return optim

    def training_step(self, batch, batch_idx):
        
        sdf_pred, grads_pred, sdf_target, normals_target = self.forward(batch)
        loss, sdf_loss, normal_loss, grad_norm_loss, sdf_on = self.compute_losses(sdf_pred, grads_pred, sdf_target, normals_target)
        self.log('train_loss', loss)
        self.log('sdf_loss', sdf_loss)
        self.log('normal_loss', normal_loss)
        self.log('grad_norm_loss', grad_norm_loss)
        self.log('sdf_on_surface', sdf_on)
        #self.log('normal_on_surface', normal_on)

        return loss
    
class DeformationSDF(pl.LightningModule):
    def __init__(self, lr=0.0001, pretrained_sdf_path=None, classifier_net_path=None, grad_a=5e1, normal_constraint_a=1e2, sdf_a=1e1, exp_penalty_a=1., exp_scale=1e2):
        super(DeformationSDF, self).__init__()
        self.save_hyperparameters()

    
        pretrained_sdf = VSM.load_from_checkpoint(pretrained_sdf_path).model

        self.sdf_net = pretrained_sdf
        self.sdf_net.eval()
        # freeze the parameters
        for param in self.sdf_net.parameters():
            param.requires_grad = False

        pretrained_classifier = ClassificationVSM.load_from_checkpoint(classifier_net_path).model
        self.classifier_net = pretrained_classifier
        self.classifier_net.eval()
        # freeze the parameters
        for param in self.classifier_net.parameters():
            param.requires_grad = False

        self.bl_transform = TransformationSiren()
        
        self.br_transform = TransformationSiren()
        
        self.fr_transform = TransformationSiren()

        self.fl_transform = TransformationSiren()
        
        # each transformation has 7 parameters (3 for translation, 4 for rotation)

        state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
        state = state[7:]
        mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
        joint_limits = mjmodel.jnt_range[1:, :]
        state = 2*(state - joint_limits[:, 0])/(joint_limits[:, 1] - joint_limits[:, 0]) - 1
        self.template_state = state


        class_files = [f for f in os.listdir("/home/sammoore/Documents/PointCloud-PointForce/classification_data") if f.startswith('rigid_link_pc_0')]

        # make dictionary of class names using the end of the file name (after rigid_link_pc_0_)
        self.class_names = {}
        for i, file in enumerate(class_files):
            class_name = file.split('rigid_link_pc_0_')[1].split('.')[0]
            self.class_names[class_name] = i + 1
        
        print(self.class_names)

        self.br_classes = {}
        # if BR is in key, then then assign to the BR class
        for key, value in self.class_names.items():
            if 'BR' in key:
                self.br_classes[key] = value
        
        self.bl_classes = {}
        for key, value in self.class_names.items():
            if 'BL' in key:
                self.bl_classes[key] = value
        
        self.fr_classes = {}
        for key, value in self.class_names.items():
            if 'FR' in key:
                self.fr_classes[key] = value
        
        self.fl_classes = {}
        for key, value in self.class_names.items():
            if 'FL' in key:
                self.fl_classes[key] = value


        self.grad_a = grad_a
        self.normal_constraint_a = normal_constraint_a
        self.sdf_a = sdf_a
        self.exp_penalty_a = exp_penalty_a
        self.exp_scale = exp_scale
        self.lr = lr

    def template(self, xyzs, states):

        input = torch.cat((xyzs, states), dim=1)
        sdf_old = self.sdf_net(input)
        return sdf_old
    
    def forward(self, x):
        xyz_A = x['xyz_A']
        sdf_A = x['sdf_A']
        classes_A = x['classes_A']
        xyz_B = x['xyz_B']
        sdf_B = x['sdf_B']
        classes_B = x['classes_B']
        state = x['joint_data']
        off_xyz = x['off_xyz']
        off_sdf = x['off_sdf']
        off_classes = x['off_classes']

        xyz_A = torch.reshape(xyz_A, (-1, 3))
        sdf_A = torch.reshape(sdf_A, (-1, 1))
        classes_A = torch.reshape(classes_A, (-1, 1))
        xyz_B = torch.reshape(xyz_B, (-1, 3))
        sdf_B = torch.reshape(sdf_B, (-1, 1))
        classes_B = torch.reshape(classes_B, (-1, 1))
    
        state = torch.reshape(state, (-1, state.shape[2]))

        off_xyz = torch.reshape(off_xyz, (-1, 3))
        off_sdf = torch.reshape(off_sdf, (-1, 1))
        off_classes = torch.reshape(off_classes, (-1, 1))

        dist_A_to_B = torch.norm(xyz_A - xyz_B, dim=1)
        dist_A_to_B = torch.reshape(dist_A_to_B, (-1, 1))

        on_xyz = torch.cat((xyz_A, xyz_B))
        on_sdf = torch.cat((sdf_A, sdf_B))

        xyzs = torch.cat((on_xyz, off_xyz))
        sdf = torch.cat((on_sdf, off_sdf))
        states = torch.cat((state, state)).requires_grad_(True)

        template_states = np.repeat(self.template_state[np.newaxis, :], states.shape[0], axis=0)
        template_states = torch.from_numpy(template_states).float().to('cuda')

        classes = torch.cat((classes_A, classes_B))
        classes = torch.cat((classes, off_classes))

        # get indices of the BR, BL, FR, FL classes
        br_class_idx = [self.class_names[key] for key in self.br_classes.keys()]
        bl_class_idx = [self.class_names[key] for key in self.bl_classes.keys()]
        fr_class_idx = [self.class_names[key] for key in self.fr_classes.keys()]
        fl_class_idx = [self.class_names[key] for key in self.fl_classes.keys()]

        # get the indices of the classes in the batch
        br_indices = torch.where(classes == br_class_idx[0], torch.ones_like(classes), torch.zeros_like(classes))
        br_indices = torch.where(classes == br_class_idx[1], torch.ones_like(classes), br_indices)
        br_indices = torch.where(classes == br_class_idx[2], torch.ones_like(classes), br_indices)

        bl_indices = torch.where(classes == bl_class_idx[0], torch.ones_like(classes), torch.zeros_like(classes))
        bl_indices = torch.where(classes == bl_class_idx[1], torch.ones_like(classes), bl_indices)
        bl_indices = torch.where(classes == bl_class_idx[2], torch.ones_like(classes), bl_indices)

        fr_indices = torch.where(classes == fr_class_idx[0], torch.ones_like(classes), torch.zeros_like(classes))
        fr_indices = torch.where(classes == fr_class_idx[1], torch.ones_like(classes), fr_indices)
        fr_indices = torch.where(classes == fr_class_idx[2], torch.ones_like(classes), fr_indices)

        fl_indices = torch.where(classes == fl_class_idx[0], torch.ones_like(classes), torch.zeros_like(classes))
        fl_indices = torch.where(classes == fl_class_idx[1], torch.ones_like(classes), fl_indices)
        fl_indices = torch.where(classes == fl_class_idx[2], torch.ones_like(classes), fl_indices)

        br_xyz = xyzs[br_indices.squeeze() == 1]
        bl_xyz = xyzs[bl_indices.squeeze() == 1]
        fr_xyz = xyzs[fr_indices.squeeze() == 1]
        fl_xyz = xyzs[fl_indices.squeeze() == 1]

        # np.array([FL-Hip-ABAD, FL-Hip-FLEX, FL-Knee, FR-Hip-ABAD, FR-Hip-FLEX, FR-Knee, BL-Hip-ABAD, BL-Hip-FLEX, BL-Knee, BR-Hip-ABAD, BR-Hip-FLEX, BR-Knee], dtype=np.float64) # robot POV
        br_states = states[br_indices.squeeze() == 1]
        br_states = br_states[:, 9:]
        bl_states = states[bl_indices.squeeze() == 1]
        bl_states = bl_states[:, 6:9]
        fr_states = states[fr_indices.squeeze() == 1]
        fr_states = fr_states[:, 3:6]
        fl_states = states[fl_indices.squeeze() == 1]
        fl_states = fl_states[:, :3]

        # remap the classes to 0, 1, 2 for each leg
        br_classes = torch.where(classes == br_class_idx[0], torch.zeros_like(classes), torch.zeros_like(classes))
        br_classes = torch.where(classes == br_class_idx[1], torch.ones_like(classes), br_classes)
        br_classes = torch.where(classes == br_class_idx[2], 2*torch.ones_like(classes), br_classes)

        bl_classes = torch.where(classes == bl_class_idx[0], torch.zeros_like(classes), torch.zeros_like(classes))
        bl_classes = torch.where(classes == bl_class_idx[1], torch.ones_like(classes), bl_classes)
        bl_classes = torch.where(classes == bl_class_idx[2], 2*torch.ones_like(classes), bl_classes)

        fr_classes = torch.where(classes == fr_class_idx[0], torch.zeros_like(classes), torch.zeros_like(classes))
        fr_classes = torch.where(classes == fr_class_idx[1], torch.ones_like(classes), fr_classes)
        fr_classes = torch.where(classes == fr_class_idx[2], 2*torch.ones_like(classes), fr_classes)

        fl_classes = torch.where(classes == fl_class_idx[0], torch.zeros_like(classes), torch.zeros_like(classes))
        fl_classes = torch.where(classes == fl_class_idx[1], torch.ones_like(classes), fl_classes)
        fl_classes = torch.where(classes == fl_class_idx[2], 2*torch.ones_like(classes), fl_classes)

        br_new_classes  = br_classes[br_indices.squeeze() == 1]
        bl_new_classes  = bl_classes[bl_indices.squeeze() == 1]
        fr_new_classes  = fr_classes[fr_indices.squeeze() == 1]
        fl_new_classes  = fl_classes[fl_indices.squeeze() == 1]

        br_one_hot = F.one_hot(br_new_classes.to(torch.int64), num_classes=3)
        bl_one_hot = F.one_hot(bl_new_classes.to(torch.int64), num_classes=3)
        fr_one_hot = F.one_hot(fr_new_classes.to(torch.int64), num_classes=3)
        fl_one_hot = F.one_hot(fl_new_classes.to(torch.int64), num_classes=3)

        #print(br_one_hot)
        
        #print(torch.unique(br_new_classes))
        #print(br_states.shape)
        br_states = br_states.requires_grad_(True)
        bl_states = bl_states.requires_grad_(True)
        fr_states = fr_states.requires_grad_(True)
        fl_states = fl_states.requires_grad_(True)
        br_input = torch.cat((br_states, br_one_hot.squeeze(1)), dim=1)
        bl_input = torch.cat((bl_states, bl_one_hot.squeeze(1)), dim=1)
        fr_input = torch.cat((fr_states, fr_one_hot.squeeze(1)), dim=1)
        fl_input = torch.cat((fl_states, fl_one_hot.squeeze(1)), dim=1)

        br_template_state = template_states[:, 9:]
        bl_template_state = template_states[:, 6:9]
        fr_template_state = template_states[:, 3:6]
        fl_template_state = template_states[:, :3]

        br_template_state = br_template_state[br_indices.squeeze() == 1]
        bl_template_state = bl_template_state[bl_indices.squeeze() == 1]
        fr_template_state = fr_template_state[fr_indices.squeeze() == 1]
        fl_template_state = fl_template_state[fl_indices.squeeze() == 1]

        br_input_null = torch.cat((br_template_state, br_one_hot.squeeze(1)), dim=1)
        bl_input_null = torch.cat((bl_template_state, bl_one_hot.squeeze(1)), dim=1)
        fr_input_null = torch.cat((fr_template_state, fr_one_hot.squeeze(1)), dim=1)
        fl_input_null = torch.cat((fl_template_state, fl_one_hot.squeeze(1)), dim=1)


        br_pose = self.br_transform(br_input)
        bl_pose = self.bl_transform(bl_input)
        fr_pose = self.fr_transform(fr_input)
        fl_pose = self.fl_transform(fl_input)
        br_grads = torch.autograd.grad(br_pose, br_states, grad_outputs=torch.ones_like(br_pose), create_graph=True)[0]
        bl_grads = torch.autograd.grad(bl_pose, bl_states, grad_outputs=torch.ones_like(bl_pose), create_graph=True)[0]
        fr_grads = torch.autograd.grad(fr_pose, fr_states, grad_outputs=torch.ones_like(fr_pose), create_graph=True)[0]
        fl_grads = torch.autograd.grad(fl_pose, fl_states, grad_outputs=torch.ones_like(fl_pose), create_graph=True)[0]

        grads = torch.cat((br_grads, bl_grads, fr_grads, fl_grads))
        # make sure the change in pose is smooth
        grad_norm_loss = torch.abs(grads.norm(dim=1)).mean()

        br_pose_null = self.br_transform(br_input_null)
        bl_pose_null = self.bl_transform(bl_input_null)
        fr_pose_null = self.fr_transform(fr_input_null)
        fl_pose_null = self.fl_transform(fl_input_null)


        br_xyz_new = self.transform_points(br_xyz, br_pose)
        bl_xyz_new = self.transform_points(bl_xyz, bl_pose)
        fr_xyz_new = self.transform_points(fr_xyz, fr_pose)
        fl_xyz_new = self.transform_points(fl_xyz, fl_pose)

        br_xyz_new_null = self.transform_points(br_xyz, br_pose_null)
        bl_xyz_new_null = self.transform_points(bl_xyz, bl_pose_null)
        fr_xyz_new_null = self.transform_points(fr_xyz, fr_pose_null)
        fl_xyz_new_null = self.transform_points(fl_xyz, fl_pose_null)


        new_xyz = torch.cat((br_xyz_new, bl_xyz_new, fr_xyz_new, fl_xyz_new))
        new_xyz_null = torch.cat((br_xyz_new_null, bl_xyz_new_null, fr_xyz_new_null, fl_xyz_new_null))
        xyz = torch.cat((br_xyz, bl_xyz, fr_xyz, fl_xyz))
        delta_xyz_null = torch.norm(new_xyz - new_xyz_null, dim=1).unsqueeze(-1)

        # get the original classes
        br_classes = classes[br_indices.squeeze() == 1]
        bl_classes = classes[bl_indices.squeeze() == 1]
        fr_classes = classes[fr_indices.squeeze() == 1]
        fl_classes = classes[fl_indices.squeeze() == 1]

        classes_legs = torch.cat((br_classes, bl_classes, fr_classes, fl_classes))
        classes_legs_hat = self.classifier_net(new_xyz)

        br_sdf = sdf[br_indices.squeeze() == 1]
        bl_sdf = sdf[bl_indices.squeeze() == 1]
        fr_sdf = sdf[fr_indices.squeeze() == 1]
        fl_sdf = sdf[fl_indices.squeeze() == 1]

        sdf_legs = torch.cat((br_sdf, bl_sdf, fr_sdf, fl_sdf))
        sdf_legs_hat = self.template(new_xyz, template_states[:new_xyz.shape[0], :])

        #input = torch.cat((xyzs, states), dim=1)
        #delta_xyz = self.deformation_net(input)
        #new_xyz = xyzs + delta_xyz
        #sdf_hat = self.template(new_xyz, template_states)
        
        # if the state is the same as the template state, the deformation should be null
        #input_null = torch.cat((xyzs, template_states), dim=1)
        #delta_xyz_null = self.deformation_net(input_null)



        #new_xyz_A = new_xyz[:xyz_A.shape[0]]
        #new_xyz_B = new_xyz[xyz_A.shape[0]:xyz_A.shape[0]+xyz_B.shape[0]]

        # because the transformation is rigid, the distance between the points on the rigid bodies should not change
        #dist_A_to_B_new = torch.norm(new_xyz_A - new_xyz_B, dim=1).unsqueeze(-1)

        

        return (sdf_legs, sdf_legs_hat, classes_legs, classes_legs_hat, delta_xyz_null, grad_norm_loss)



    def compute_losses(self, sdf, sdf_hat, classes, classes_hat, delta_xyz_null, grad_norm_loss):
        
        sdf_loss = torch.where(sdf == 0, torch.abs(sdf_hat), torch.zeros_like(sdf_hat)).mean()

        #exp_penalty =  torch.where(sdf==0, torch.zeros_like(sdf_hat), torch.exp(-self.exp_scale*torch.abs(sdf_hat))).mean()
        #smoothness = torch.abs(deform_grads.norm(dim=1)).mean()
        delta_xyz_null_loss = torch.abs(delta_xyz_null).mean()
        # the distance between the points on the rigid bodies should not change
        #dist_loss = F.l1_loss(dist_A_to_B, dist_A_to_B_new)
        weights = torch.ones(14, dtype=torch.float32).to('cuda')
        weights[13] = 0.064
        weights[0] = 0.2759
        classification_loss = F.cross_entropy(classes_hat, classes.squeeze(-1), weight=weights)


        total_loss =  sdf_loss*1e2 + classification_loss*1 + delta_xyz_null_loss*1e2 + grad_norm_loss*0

        return total_loss, sdf_loss, classification_loss, delta_xyz_null_loss, grad_norm_loss

    def configure_optimizers(self):
        optim = torch.optim.Adam([
        {"params": self.fl_transform.parameters(), "lr": self.lr},
        {"params": self.fr_transform.parameters(), "lr": self.lr},
        {"params": self.bl_transform.parameters(), "lr": self.lr},
        {"params": self.br_transform.parameters(), "lr": self.lr}])
        return optim

    def training_step(self, batch, batch_idx):
        
        output = self.forward(batch)
        loss, sdf_loss, classification_loss, delta_xyz, grad_norm_loss = self.compute_losses(*output)

        self.log('train_loss', loss)
        self.log('sdf_loss', sdf_loss)
        self.log('classification_loss', classification_loss)
        self.log('delta_xyz_null', delta_xyz)
        self.log('grad_norm_loss', grad_norm_loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss, sdf_loss, classification_loss, delta_xyz, grad_norm_loss = self.compute_losses(*output)

        self.log('val_loss', loss)
        self.log('sdf_loss', sdf_loss)
        self.log('classification_loss', classification_loss)
        self.log('delta_xyz_null', delta_xyz)
        self.log('grad_norm_loss', grad_norm_loss)

        return loss

    def quaternion_to_rotation_matrix(self, quaternions):
        quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        Rxx, Rxy, Rxz = 1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)
        Ryx, Ryy, Ryz = 2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)
        Rzx, Rzy, Rzz = 2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)

        return torch.stack((
            torch.stack((Rxx, Rxy, Rxz), dim=-1),
            torch.stack((Ryx, Ryy, Ryz), dim=-1),
            torch.stack((Rzx, Rzy, Rzz), dim=-1),
        ), dim=-2)

    def transform_points(self, xyzs, transforms):
        # Assumes transforms is (batch_size, 7) where the first three are translation and last four are quaternion components
        translations = transforms[:, :3]  # shape (batch_size, 3)
        quaternions = transforms[:, 3:]   # shape (batch_size, 4)

        # Convert quaternions to rotation matrices
        rotation_matrices = self.quaternion_to_rotation_matrix(quaternions)  # shape (batch_size, 3, 3)

        # Transform points
        # xyzs needs to be (batch_size, 3, 1) for matrix multiplication
        xyzs = xyzs.unsqueeze(-1)  # Adds a new dimension for matrix multiplication
        transformed_xyzs = torch.matmul(rotation_matrices, xyzs).squeeze(-1) + translations

        return transformed_xyzs


class ClassificationVSM(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super(ClassificationVSM, self).__init__()
        self.save_hyperparameters()

    
        self.model = nn.Sequential(SirenLayer(3, 256, is_first=True),
                                SirenLayer(256, 256),
                                SirenLayer(256, 256),
                                SirenLayer(256, 256),
                                SirenLayer(256, 14, is_last=True))

        # self.model = nn.Sequential(nn.Linear(3, 256),
        #                         nn.ReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.ReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.ReLU(),
        #                         nn.Linear(256, 256),
        #                         nn.ReLU(),
        #                         nn.Linear(256, 13))
        

        self.lr = lr

    def forward(self, x):
        xyz = x['xyzs']
        classes = x['classes']
        classes = classes.squeeze(-1)
        classes_hat = self.model(xyz)
        #class_weights = torch.ones_like(classes[0, :], dtype=torch.float32)
        #class_weights[13] = 0.5
        #classes = torch.where(classes != 0, torch.ones_like(classes), classes)
        loss = F.cross_entropy(classes_hat, classes) # , weight=class_weights)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam([
        {"params": self.model.parameters(), "lr": self.lr}])
        return optim
    
    def training_step(self, batch, batch_idx):
            
            loss = self.forward(batch)
            self.log('train_loss', loss)
    
            return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('val_loss', loss)
        return loss
    

class CorrespondenceVSM(pl.LightningModule):
    def __init__(self, lr=0.0001, data_dir=None):
        super(CorrespondenceVSM, self).__init__()
        self.save_hyperparameters()

        self.mjmodel = mujoco.MjModel.from_xml_path("./Robots/unitree_go2/scene.xml")
    
        self.model = StateConditionMLPQueryModelSplit(n_states=12, hidden_features=256)
        self.model.layer6 = SirenLayer(256, 3, is_last=True)

        self.model_inv = StateConditionMLPQueryModelSplit(n_states=12, hidden_features=256)
        self.model_inv.layer6 = SirenLayer(256, 3, is_last=True)

        self.lr = lr

        # load json file with the color mapping
        with open(data_dir + '/color_map.json') as f:
            self.name2color = json.load(f)

        self.color2name = {tuple(color): id for id, color in self.name2color.items()}

        state = np.array([0, 35, -75, 0, 35, -75, 0, 35, -75, 0, 35, -75], dtype=np.float64)*(np.pi/180)
        self.template_state = state

        self.fl_hip_idx = (self.mjmodel.joint(f'FL_hip_joint').qposadr - 7).item()
        self.fl_thigh_idx = (self.mjmodel.joint(f'FL_thigh_joint').qposadr - 7).item()
        self.fl_calf_idx = (self.mjmodel.joint(f'FL_calf_joint').qposadr - 7).item()

        self.fr_hip_idx = (self.mjmodel.joint(f'FR_hip_joint').qposadr - 7).item()
        self.fr_thigh_idx = (self.mjmodel.joint(f'FR_thigh_joint').qposadr - 7).item()
        self.fr_calf_idx = (self.mjmodel.joint(f'FR_calf_joint').qposadr - 7).item()

        self.rl_hip_idx = (self.mjmodel.joint(f'RL_hip_joint').qposadr - 7).item()
        self.rl_thigh_idx = (self.mjmodel.joint(f'RL_thigh_joint').qposadr - 7).item()
        self.rl_calf_idx = (self.mjmodel.joint(f'RL_calf_joint').qposadr - 7).item()

        self.rr_hip_idx = (self.mjmodel.joint(f'RR_hip_joint').qposadr - 7).item()
        self.rr_thigh_idx = (self.mjmodel.joint(f'RR_thigh_joint').qposadr - 7).item()
        self.rr_calf_idx = (self.mjmodel.joint(f'RR_calf_joint').qposadr - 7).item()

        self.fl_hip_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FL_hip').id], dtype=torch.float).to('cuda')
        self.fl_thigh_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FL_thigh').id], dtype=torch.float).to('cuda')
        self.fl_calf_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FL_calf').id], dtype=torch.float).to('cuda')

        self.fr_hip_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FR_hip').id], dtype=torch.float).to('cuda')
        self.fr_thigh_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FR_thigh').id], dtype=torch.float).to('cuda')
        self.fr_calf_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'FR_calf').id], dtype=torch.float).to('cuda')

        self.rl_hip_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RL_hip').id], dtype=torch.float).to('cuda')
        self.rl_thigh_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RL_thigh').id], dtype=torch.float).to('cuda')
        self.rl_calf_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RL_calf').id], dtype=torch.float).to('cuda')

        self.rr_hip_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RR_hip').id], dtype=torch.float).to('cuda')
        self.rr_thigh_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RR_thigh').id], dtype=torch.float).to('cuda')
        self.rr_calf_t = torch.tensor(self.mjmodel.body_pos[self.mjmodel.body(f'RR_calf').id], dtype=torch.float).to('cuda')

    def training_step(self, batch, batch_idx):
            
            xyz_hat, xyz_template, xyz, xyz_inv = self.forward(batch)
            loss_0 = torch.abs(xyz_hat - xyz_template)
            loss_1 = torch.abs(xyz_inv - xyz)
            loss = loss_0 + loss_1
            loss = loss.mean()
            self.log('forward', loss_0.mean())
            self.log('inverse', loss_1.mean())
            self.log('train_loss', loss)
            return loss
    
    def validation_step(self, batch, batch_idx):
        xyz_hat, xyz_template = self.forward(batch)
        loss = F.l1_loss(xyz_hat, xyz_template)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam([
        {"params": self.model.parameters(), "lr": self.lr},
        {"params": self.model_inv.parameters(), "lr": self.lr}])
        return optim

    def forward(self, x):
        xyz = x['xyz']
        states = x['joint_data_rad']
        states_scaled = x['joint_data_scaled']
        colors = x['colors']

        xyz = torch.reshape(xyz, (-1, 3))
        states = torch.reshape(states, (-1, states.shape[2]))
        states_scaled = torch.reshape(states_scaled, (-1, states_scaled.shape[2]))
        colors = torch.reshape(colors, (-1, 3))

        colors = torch.floor(colors * 10) / 10

        homogenous_xyz = torch.cat((xyz, torch.ones_like(xyz[:, 0]).unsqueeze(-1)), dim=1)

        rr_colors = [self.name2color['RR_hip'], self.name2color['RR_thigh'], self.name2color['RR_calf']]
        rl_colors = [self.name2color['RL_hip'], self.name2color['RL_thigh'], self.name2color['RL_calf']]
        fr_colors = [self.name2color['FR_hip'], self.name2color['FR_thigh'], self.name2color['FR_calf']]
        fl_colors = [self.name2color['FL_hip'], self.name2color['FL_thigh'], self.name2color['FL_calf']]

        xyz_local = self.transform_points(homogenous_xyz, states, colors, rr_colors, leg='rr')
        xyz_local = self.transform_points(xyz_local, states, colors, rl_colors, leg='rl')
        xyz_local = self.transform_points(xyz_local, states, colors, fr_colors, leg='fr')
        xyz_local = self.transform_points(xyz_local, states, colors, fl_colors, leg='fl')

        template_states = np.repeat(self.template_state[np.newaxis, :], states.shape[0], axis=0)
        template_states = torch.from_numpy(template_states).float().to('cuda')

        xyz_template = self.transform_points(xyz_local, template_states, colors, rr_colors, inverse=False, leg='rr')
        xyz_template = self.transform_points(xyz_template, template_states, colors, rl_colors, inverse=False, leg='rl')
        xyz_template = self.transform_points(xyz_template, template_states, colors, fr_colors, inverse=False, leg='fr')
        xyz_template = self.transform_points(xyz_template, template_states, colors, fl_colors, inverse=False, leg='fl')
        xyz_template = xyz_template[:, :3]

        input = torch.cat((xyz, states_scaled), dim=1)
        xyz_hat = xyz + self.model(input)

        input_inv = torch.cat((xyz_template, states_scaled), dim=1)
        xyz_inv = xyz_template + self.model_inv(input_inv)
        
        return xyz_hat, xyz_template, xyz, xyz_inv

    def transform_points(self, homogenous_xyz, states, colors, colors_to_transform, inverse=True, leg=None):
        if leg == 'rr':
            hip_idx = self.rr_hip_idx
            thigh_idx = self.rr_thigh_idx
            calf_idx = self.rr_calf_idx
            hip_t = self.rr_hip_t
            thigh_t = self.rr_thigh_t
            calf_t = self.rr_calf_t
        elif leg == 'rl':
            hip_idx = self.rl_hip_idx
            thigh_idx = self.rl_thigh_idx
            calf_idx = self.rl_calf_idx
            hip_t = self.rl_hip_t
            thigh_t = self.rl_thigh_t
            calf_t = self.rl_calf_t

        elif leg == 'fr':
            hip_idx = self.fr_hip_idx
            thigh_idx = self.fr_thigh_idx
            calf_idx = self.fr_calf_idx
            hip_t = self.fr_hip_t
            thigh_t = self.fr_thigh_t
            calf_t = self.fr_calf_t

        elif leg == 'fl':
            hip_idx = self.fl_hip_idx
            thigh_idx = self.fl_thigh_idx
            calf_idx = self.fl_calf_idx
            hip_t = self.fl_hip_t
            thigh_t = self.fl_thigh_t
            calf_t = self.fl_calf_t
        else:
            raise ValueError('Leg not specified')
        
        indices = self.get_indices(colors, colors_to_transform)

        if inverse: 

            R = self.R_x(states[:, hip_idx])
            T, T_inv = self.T_and_inverse(R, hip_t)

            xyz_local = torch.where((indices.squeeze() == 1).unsqueeze(-1), torch.bmm(T_inv, homogenous_xyz.unsqueeze(-1)).squeeze(-1), homogenous_xyz)

            R = self.R_y(states[:, thigh_idx])
            T, T_inv = self.T_and_inverse(R, thigh_t)

            indices = self.get_indices(colors, colors_to_transform[1:])
            xyz_local = torch.where((indices.squeeze() == 1).unsqueeze(-1), torch.bmm(T_inv, xyz_local.unsqueeze(-1)).squeeze(-1), xyz_local)

            R = self.R_y(states[:, calf_idx])
            T, T_inv = self.T_and_inverse(R, calf_t)

            indices = self.get_indices(colors, colors_to_transform[2:])
            xyz_local = torch.where((indices.squeeze() == 1).unsqueeze(-1), torch.bmm(T_inv, xyz_local.unsqueeze(-1)).squeeze(-1), xyz_local)
        
        else:

            R = self.R_y(states[:, calf_idx])
            T, T_inv = self.T_and_inverse(R, calf_t)

            indices = self.get_indices(colors, colors_to_transform[2:])
            xyz_local = torch.where(((indices.squeeze() == 1)).unsqueeze(-1), torch.bmm(T, homogenous_xyz.unsqueeze(-1)).squeeze(-1), homogenous_xyz)

            R = self.R_y(states[:, thigh_idx])
            T, T_inv = self.T_and_inverse(R, thigh_t)

            indices = self.get_indices(colors, colors_to_transform[1:])
            xyz_local = torch.where((indices.squeeze() == 1).unsqueeze(-1), torch.bmm(T, xyz_local.unsqueeze(-1)).squeeze(-1), xyz_local)

            R = self.R_x(states[:, hip_idx])
            T, T_inv = self.T_and_inverse(R, hip_t)

            indices = self.get_indices(colors, colors_to_transform)
            xyz_local = torch.where((indices.squeeze() == 1).unsqueeze(-1), torch.bmm(T, xyz_local.unsqueeze(-1)).squeeze(-1), xyz_local)

        return xyz_local

    def get_indices(self, colors, color_list):
        # Ensure colors and color_list are tensors and properly broadcasted
        color = torch.tensor(color_list[0], dtype=colors.dtype, device=colors.device)
        mask = (colors == color).all(dim=1)  # Assuming color is like [R, G, B] and colors is [N, 3]

        # Create the initial mask
        result_mask = torch.where(mask, torch.ones_like(mask, dtype=colors.dtype), torch.zeros_like(mask, dtype=colors.dtype))

        # Process remaining colors in the list, if any
        for next_color in color_list[1:]:
            next_color = torch.tensor(next_color, dtype=colors.dtype, device=colors.device)
            mask = (colors == next_color).all(dim=1)
            # Update result_mask for any additional matches
            result_mask = torch.where(mask, torch.ones_like(mask, dtype=colors.dtype), result_mask)

        return result_mask
    
    def R_x(self, thetas):
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        
        # Prepare the batch of rotation matrices
        zeros = torch.zeros_like(thetas)
        ones = torch.ones_like(thetas)
        
        # Create a batch of 3x3 rotation matrices
        rotation_matrices = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_thetas, -sin_thetas], dim=-1),
            torch.stack([zeros, sin_thetas, cos_thetas], dim=-1)
        ], dim=-2)
        
        return rotation_matrices
    
    def R_y(self, thetas):
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        
        # Prepare the batch of rotation matrices
        zeros = torch.zeros_like(thetas)
        ones = torch.ones_like(thetas)
        
        # Create a batch of 3x3 rotation matrices
        rotation_matrices = torch.stack([
            torch.stack([cos_thetas, zeros, sin_thetas], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_thetas, zeros, cos_thetas], dim=-1)
        ], dim=-2)
        
        return rotation_matrices
    
    def T_and_inverse(self, R, t):
        # R is expected to be of shape [batch_size, 3, 3]
        # t is expected to be of shape [3]
        
        batch_size = R.shape[0]
        
        # Initialize the transformation matrices with batched identity matrices
        out = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Insert the rotation matrices
        out[:, :3, :3] = R
        
        # Repeat the translation vector for each element in the batch and set it
        out[:, :3, 3] = t.unsqueeze(0).repeat(batch_size, 1)
        
        # Compute the inverse of each transformation matrix in the batch
        # Since T is a rigid transformation, the inverse is easy to calculate:
        # If T = [ R  t ]
        #         [ 0  1 ]
        # Then T^-1 = [ R^T  -R^T*t ]
        #             [  0        1  ]
        # Compute transpose of R (rotation matrices are orthogonal)
        R_transpose = R.transpose(1, 2)
        # Compute -R^T * t
        neg_Rt_t = -torch.matmul(R_transpose, t.view(3, 1)).squeeze(-1)
        
        # Initialize the inverse transformation matrices
        T_inv = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        T_inv[:, :3, :3] = R_transpose
        T_inv[:, :3, 3] = neg_Rt_t
        
        return out.to('cuda'), T_inv.to('cuda')

    
if __name__ == '__main__':    

    sdf_a = 3e3 #np.random.uniform(3e3, 8e3)
    grad_a = 5e1 #np.random.uniform(5e1, 1e2)
    exp_penalty_a = 1e2
    exp_scale = 1e2
    lr = 5e-5 #1e-4 #1e-5 #1e-4 #1.5e-5 #1e-5
    epochs = 400
    n_obs = 102400 #13208
    devices = 1
    batch_size = 64 #384
    model =  VSM(lr=lr, exp_penalty_a=exp_penalty_a, exp_scale=exp_scale, grad_a=grad_a, sdf_a=sdf_a, epochs=epochs, n_obs=n_obs, devices=devices, batch_size=batch_size)
    #model.model.stage1()
    #model = VSM.load_from_checkpoint(f'./lightning_logs/version_172/checkpoints/newest.ckpt').to('cpu') 
    #model.lr = lr
    #model.batch_size = batch_size
    #model.exp_scale = exp_scale
    #model.devices = devices
    #model = VSM()
    data_dir = './Balanced_Data'
    dm = VSMDataModule(data_dir, batch_size=batch_size, num_workers=8, split_ratio=(1, 0, 0))
    dm.prepare_data()
    dm.setup()

    # testing the training loop
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='newest', every_n_train_steps=40, enable_version_counter=False)
    annealing_callback = CyclicalAnnealingCallback(max_epochs=epochs, steps_per_epoch=int(np.ceil(n_obs*(1/batch_size)*(1/devices))), pretrain_epochs=5, type='linear') #29583 #5917 #925
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback, annealing_callback], devices=devices, strategy='auto')
    #trainer = pl.Trainer()
    #trainer.fit(model, dm)
    #tuner = pl.tuner.tuning.Tuner(trainer)
    #lr_finder = tuner.lr_find(model, dm)
    #
    #print(lr_finder.suggestion())
    # plot lr finder
    #fig = lr_finder.plot(suggest=True)
    #fig.set_xlim(1e-4, 1e-3)
    #fig.savefig(f'{save_dir}/version_{i}/lr_finder.png')
    #fig.show()
    trainer.fit(model, dm)

    pretrained = VSM.load_from_checkpoint(f'/home/sammoore/Documents/PointCloud-PointForce/lightning_logs/version_137/checkpoints/newest.ckpt').to('cuda')
    pretrained_model = pretrained.model
    model = BlendedSDF(pretrained_sdf=pretrained_model, n_states=12, lr_pretrained=1e-6, lr_switch_net=5e-5)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[checkpoint_callback], devices=devices, strategy='auto')
    #trainer.fit(model, dm)
