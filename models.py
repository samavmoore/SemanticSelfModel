
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# np.array([x0, x1, x2, theta0, theta1, theta2, random, FL-Hip-ABAD, FL-Hip-FLEX, FL-Knee, FR-Hip-ABAD, FR-Hip-FLEX, FR-Knee, BL-Hip-ABAD, BL-Hip-FLEX, BL-Knee, BR-Hip-ABAD, BR-Hip-FLEX, BR-Knee], dtype=np.float64) # robot POV
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)
    
class TransformationSiren(nn.Module):
    def __init__(self, in_channels=6, out_channels=7, hidden_features=256):
        super(TransformationSiren, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layerq2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layerc1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layerc2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def forward(self, x):
        q = x[:, :3]
        c = x[:, 3:]
        q = self.layerq1(q)
        q = self.layerq2(q)
        c = self.layerc1(c)
        c = self.layerc2(c)
        x = torch.cat((q, c), dim=1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class StateConditionMLPQueryModel(torch.nn.Module):
    def __init__(self, in_channels=4, out_channels=1, hidden_features=256):
        super(StateConditionMLPQueryModel, self).__init__()

        half_hidden_features = int(hidden_features / 2)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layers1 = SirenLayer(in_channels-3, half_hidden_features, is_first=True)
        self.layers2 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers3 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layers4 = SirenLayer(half_hidden_features, half_hidden_features)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, out_channels, is_last=True)
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x

    def state_encoder(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        return x

    def forward(self, x):
        state = x[:, 3:]
        xyz = x[:, :3]
        query_feat = self.query_encoder(x[:, :3])
        state_feat = self.state_encoder(x[:, 3:])
        x = torch.cat((query_feat, state_feat), dim=1)
        #x = torch.cat((state, x), dim=1)
        #x = torch.cat((xyz, x), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
    
class StateConditionMLPQueryModelSplit(torch.nn.Module):
    def __init__(self, n_states, hidden_features=512):
        super(StateConditionMLPQueryModelSplit, self).__init__()

        names = ['FL', 'FR', 'BL', 'BR']
        states_per_leg = int(n_states / 4) 

        half_hidden_features = int(hidden_features / 2)
        eighth_hidden_features = int(hidden_features / 8)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, hidden_features)
        self.layer6 = SirenLayer(hidden_features, 1, is_last=True)

        for i, name in enumerate(names):
            
            setattr(self, f'layer_{name}_{0}', SirenLayer(states_per_leg, eighth_hidden_features, is_first=True))
            setattr(self, f'layer_{name}_{1}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
            setattr(self, f'layer_{name}_{2}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
            setattr(self, f'layer_{name}_{3}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x
    
    def FL_encoder(self, x):
        x = self.layer_FL_0(x)
        x = self.layer_FL_1(x)
        x = self.layer_FL_2(x)
        x = self.layer_FL_3(x)
        return x
    
    def FR_encoder(self, x):
        x = self.layer_FR_0(x)
        x = self.layer_FR_1(x)
        x = self.layer_FR_2(x)
        x = self.layer_FR_3(x)
        return x
    
    def BL_encoder(self, x):
        x = self.layer_BL_0(x)
        x = self.layer_BL_1(x)
        x = self.layer_BL_2(x)
        x = self.layer_BL_3(x)
        return x
    
    def BR_encoder(self, x):
        x = self.layer_BR_0(x)
        x = self.layer_BR_1(x)
        x = self.layer_BR_2(x)
        x = self.layer_BR_3(x)
        return x

    def forward(self, x):
        xyz = x[:, :3]
        states = x[:, 3:]
        FL_state = states[:, :3]
        FR_state = states[:, 3:6]
        BL_state = states[:, 6:9]
        BR_state = states[:, 9:]
        query_feat = self.query_encoder(xyz)
        FL_feat = self.FL_encoder(FL_state)
        FR_feat = self.FR_encoder(FR_state)
        BL_feat = self.BL_encoder(BL_state)
        BR_feat = self.BR_encoder(BR_state)
        state_feat = torch.cat((FL_feat, FR_feat, BL_feat, BR_feat), dim=1)
        x = torch.cat((query_feat, state_feat), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class StateConditionMLPQueryModelSplitFuse(torch.nn.Module):
    def __init__(self, n_states, hidden_features=512):
        super(StateConditionMLPQueryModelSplitFuse, self).__init__()

        names = ['FL', 'FR', 'BL', 'BR']
        states_per_leg = int(n_states / 4) 

        half_hidden_features = int(hidden_features / 2)
        eighth_hidden_features = int(hidden_features / 8)
        self.layerq1 = SirenLayer(3, half_hidden_features, is_first=True)
        self.layer2 = SirenLayer(hidden_features, hidden_features)
        self.layer3 = SirenLayer(hidden_features, hidden_features)
        self.layer4 = SirenLayer(hidden_features, hidden_features)
        self.layer5 = SirenLayer(hidden_features, hidden_features)
        self.layer6 = SirenLayer(hidden_features, 1, is_last=True)
        
        self.fuse_layer = SirenLayer(half_hidden_features, half_hidden_features)

        for i, name in enumerate(names):
            
            setattr(self, f'layer_{name}_{0}', SirenLayer(states_per_leg, eighth_hidden_features, is_first=True))
            setattr(self, f'layer_{name}_{1}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
            setattr(self, f'layer_{name}_{2}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
            setattr(self, f'layer_{name}_{3}', SirenLayer(eighth_hidden_features, eighth_hidden_features))
    
    def query_encoder(self, x):
        x = self.layerq1(x)
        return x
    
    def FL_encoder(self, x):
        x = self.layer_FL_0(x)
        x = self.layer_FL_1(x)
        x = self.layer_FL_2(x)
        x = self.layer_FL_3(x)
        return x
    
    def FR_encoder(self, x):
        x = self.layer_FR_0(x)
        x = self.layer_FR_1(x)
        x = self.layer_FR_2(x)
        x = self.layer_FR_3(x)
        return x
    
    def BL_encoder(self, x):
        x = self.layer_BL_0(x)
        x = self.layer_BL_1(x)
        x = self.layer_BL_2(x)
        x = self.layer_BL_3(x)
        return x
    
    def BR_encoder(self, x):
        x = self.layer_BR_0(x)
        x = self.layer_BR_1(x)
        x = self.layer_BR_2(x)
        x = self.layer_BR_3(x)
        return x

    def forward(self, x):
        xyz = x[:, :3]
        states = x[:, 3:]
        FL_state = states[:, :3]
        FR_state = states[:, 3:6]
        BL_state = states[:, 6:9]
        BR_state = states[:, 9:]
        query_feat = self.query_encoder(xyz)
        FL_feat = self.FL_encoder(FL_state)
        FR_feat = self.FR_encoder(FR_state)
        BL_feat = self.BL_encoder(BL_state)
        BR_feat = self.BR_encoder(BR_state)
        state_feat = torch.cat((FL_feat, FR_feat, BL_feat, BR_feat), dim=1)
        state_feat = self.fuse_layer(state_feat)
        x = torch.cat((query_feat, state_feat), dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


class KinematicNetworkSiren(nn.Module):
    def __init__(self, joint_input_dim, hidden_dim, latent_dim):
        super(KinematicNetworkSiren, self).__init__()
        
        self.leg1_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg2_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg3_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg4_net = self._build_leg_net(joint_input_dim, hidden_dim)


    def _build_leg_net(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            SirenLayerMod(in_f=hidden_dim, out_f=hidden_dim, is_first=True))
        #return SirenNet(input_dim, hidden_dim, hidden_dim, 3)

    def forward(self, joint_angles):
        leg1_output = self.leg1_net(joint_angles[:, 0:3])
        leg2_output = self.leg2_net(joint_angles[:, 3:6])
        leg3_output = self.leg3_net(joint_angles[:, 6:9])
        leg4_output = self.leg4_net(joint_angles[:, 9:12])
        
        combined_output = torch.cat((leg1_output, leg2_output, leg3_output, leg4_output), dim=1)
        
        return combined_output
    
class KinematicNetworkFullSiren(nn.Module):
    def __init__(self, joint_input_dim, hidden_dim, latent_dim):
        super(KinematicNetworkFullSiren, self).__init__()
        
        self.leg1_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg2_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg3_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg4_net = self._build_leg_net(joint_input_dim, hidden_dim)


    def _build_leg_net(self, input_dim, hidden_dim):
        return nn.Sequential(
            SirenLayerMod(in_f=input_dim, out_f=hidden_dim, is_first=True),
            SirenLayerMod(in_f=hidden_dim, out_f=hidden_dim, is_first=False),
            SirenLayerMod(in_f=hidden_dim, out_f=hidden_dim, is_first=False),)
        #return SirenNet(input_dim, hidden_dim, hidden_dim, 3)

    def forward(self, joint_angles):
        leg1_output = self.leg1_net(joint_angles[:, 0:3])
        leg2_output = self.leg2_net(joint_angles[:, 3:6])
        leg3_output = self.leg3_net(joint_angles[:, 6:9])
        leg4_output = self.leg4_net(joint_angles[:, 9:12])
        
        combined_output = torch.cat((leg1_output, leg2_output, leg3_output, leg4_output), dim=1)
        
        return combined_output

class KinematicNetworkRelu(nn.Module):
    def __init__(self, joint_input_dim, hidden_dim, latent_dim):
        super(KinematicNetworkRelu, self).__init__()
        
        self.leg1_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg2_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg3_net = self._build_leg_net(joint_input_dim, hidden_dim)
        self.leg4_net = self._build_leg_net(joint_input_dim, hidden_dim)


    def _build_leg_net(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU())
        #return SirenNet(input_dim, hidden_dim, hidden_dim, 3)

    def forward(self, joint_angles):
        leg1_output = self.leg1_net(joint_angles[:, 0:3])
        leg2_output = self.leg2_net(joint_angles[:, 3:6])
        leg3_output = self.leg3_net(joint_angles[:, 6:9])
        leg4_output = self.leg4_net(joint_angles[:, 9:12])
        
        combined_output = torch.cat((leg1_output, leg2_output, leg3_output, leg4_output), dim=1)
        
        return combined_output

class ModulatorSiren(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                SirenLayerMod(in_f=dim, out_f=dim_hidden, is_first=False)
            ))
            #self.layers.append(nn.Sequential(
            #    nn.Linear(dim, dim_hidden),
            #    nn.ReLU(),
            #))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)
    
class ModulatorSirenSplit(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers1 = nn.ModuleList([])
        self.layers2 = nn.ModuleList([])
        self.layers3 = nn.ModuleList([])
        self.layers4 = nn.ModuleList([])

        dim_in = int(dim_in / 4)
        dim_hidden = int(dim_hidden / 4)
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers1.append(nn.Sequential(
                SirenLayerMod(in_f=dim, out_f=dim_hidden, is_first=False)
            ))
            self.layers2.append(nn.Sequential(
                SirenLayerMod(in_f=dim, out_f=dim_hidden, is_first=False)
            ))
            self.layers3.append(nn.Sequential(
                SirenLayerMod(in_f=dim, out_f=dim_hidden, is_first=False)
            ))
            self.layers4.append(nn.Sequential(
                SirenLayerMod(in_f=dim, out_f=dim_hidden, is_first=False)
            ))

    def forward(self, z):
        z1, z2, z3, z4 = torch.chunk(z, 4, dim=1)
        x1, x2, x3, x4 = z1, z2, z3, z4
        hiddens = []

        for layer1, layer2, layer3, layer4 in zip(self.layers1, self.layers2, self.layers3, self.layers4):
            x1 = layer1(x1)
            x2 = layer2(x2)
            x3 = layer3(x3)
            x4 = layer4(x4)
            x = torch.cat((x1, x2, x3, x4), dim=-1)
            hiddens.append(x)
            x1 = torch.cat((x1, z1), dim=-1)
            x2 = torch.cat((x2, z2), dim=-1)
            x3 = torch.cat((x3, z3), dim=-1)
            x4 = torch.cat((x4, z4), dim=-1)

        return tuple(hiddens)

class ModulatorRelu(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU(),
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=-1)

        return tuple(hiddens)

class SirenLayerMod(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, mod=None):
        x = self.linear(x)
        if mod is not None:
            x *= mod
        return x if self.is_last else torch.sin(self.w0 * x)

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30., w0_initial=30., use_bias=True, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = SirenLayerMod(
                in_f=layer_dim_in,
                out_f=dim_hidden,
                w0=layer_w0,
                is_first=is_first
            )

            self.layers.append(layer)

        
        self.last_layer = SirenLayerMod(dim_hidden, dim_out, w0=w0, is_last=True)

    def forward(self, x, mods=None):
        mods = mods if mods is not None else [None] * self.num_layers

        for layer, mod in zip(self.layers, mods):
            x = layer(x, mod)

        return self.last_layer(x)
    
class QuadrupedSDFSiren(nn.Module):
    def __init__(self, joint_input_dim, xyz_input_dim, hidden_dim, sdf_output_dim, latent_dim, num_siren_layers=6):
        super(QuadrupedSDFSiren, self).__init__()
        self.kinematic_net = KinematicNetworkSiren(joint_input_dim, int(hidden_dim/4), latent_dim)
        self.modulator = ModulatorSiren(latent_dim, hidden_dim, num_siren_layers - 1)
        self.siren_sdf_net = SirenNet(xyz_input_dim, hidden_dim, sdf_output_dim, num_siren_layers)


    def forward(self, x):
        joint_angles = x[:, 3:]
        xyz = x[:, :3]
        
        latent_code = self.kinematic_net(joint_angles)
        mods = self.modulator(latent_code)
        sdf_output = self.siren_sdf_net(xyz, mods)
        
        return sdf_output

class QuadrupedSDFRelu(nn.Module):
    def __init__(self, joint_input_dim, xyz_input_dim, hidden_dim, sdf_output_dim, latent_dim, num_siren_layers=6):
        super(QuadrupedSDFRelu, self).__init__()
        self.kinematic_net = KinematicNetworkRelu(joint_input_dim, int(hidden_dim/4), latent_dim)
        self.modulator = ModulatorRelu(latent_dim, hidden_dim, num_siren_layers - 1)
        self.siren_sdf_net = SirenNet(xyz_input_dim, hidden_dim, sdf_output_dim, num_siren_layers)


    def forward(self, x):
        joint_angles = x[:, 3:]
        xyz = x[:, :3]
        
        latent_code = self.kinematic_net(joint_angles)
        mods = self.modulator(latent_code)
        sdf_output = self.siren_sdf_net(xyz, mods)
        
        return sdf_output
    
class QuadrupedSDFSirenSplit(nn.Module):
    def __init__(self, joint_input_dim, xyz_input_dim, hidden_dim, sdf_output_dim, latent_dim, num_siren_layers=6):
        super(QuadrupedSDFSirenSplit, self).__init__()
        self.kinematic_net = KinematicNetworkSiren(joint_input_dim, int(hidden_dim/4), latent_dim)
        self.modulator = ModulatorSirenSplit(latent_dim, hidden_dim, num_siren_layers - 1)
        self.siren_sdf_net = SirenNet(xyz_input_dim, hidden_dim, sdf_output_dim, num_siren_layers)


    def forward(self, x):
        joint_angles = x[:, 3:]
        xyz = x[:, :3]
        
        latent_code = self.kinematic_net(joint_angles)
        mods = self.modulator(latent_code)
        sdf_output = self.siren_sdf_net(xyz, mods)
        
        return sdf_output
