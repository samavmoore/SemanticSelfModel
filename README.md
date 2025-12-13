# Semantic Visual Self-Modeling for Whole-Body Awareness in Legged Robots

We started this projected with the intention of extending my advisor's PhD work _Full-body visual self-modeling of robot morphologies_ ([Arxiv](https://arxiv.org/abs/2111.06389)) to legged robots. The original framework uses deformable implicit 3-D representations (DeepSDFs with siren layers) conditioned on joint-angles to model robot bodies from pointcloud observations. Then, this model (shown below) is used for a range of downstream planning tasks.

<img width="858" height="252" alt="VSM_science_robotics" src="https://github.com/user-attachments/assets/43cd4175-4c57-4177-b07c-a07c2f9796e7" />

We wanted to address a few limitations/challenges of the original work:

1. Accurate joint-angle conditioned SDFs are challenging to learn for higher DOF robots
2. It is difficult to maintain on-surface SDF resolution while making accurate off-surface distance predictions
3. SDFs only provide geometric information while additional semantic information would be
   invaluable for downstream tasks like self/world modeling and planning. 

The first thing that this code does is test all of the architectures in the figure below (a is original while others are new) on the unitree Go2 to determine their scalability to higher DOF platforms that robot arms.

<img width="654" height="600" alt="VSM_architectures" src="https://github.com/user-attachments/assets/5dfc5080-e78b-4e75-93a4-e91e37c66bd6" />


The blue layers use siren activations while gray use ReLU. The Hadamard product indicates the product of features, that almost modulates the frequencies like attention $x_{\ell+1} = \sin(W_\ell (z_{\ell} \odot x_\ell) + b_\ell)$. The models that work the best (b and e), split the leg data, $q = [q_1, q_2, q_3, q_4]$, into seperate encoders before fusing into a global representation. 

Here's an example rendering of the self-model/deformable SDF (using b) in a random configuration at test time:

https://github.com/user-attachments/assets/38ab3e2f-2121-4dac-9e59-a5e46c6570ee


The way that these models are typically trained, though, can make capturing accurate off-surface distances very challenging. Accurate signed distances far from the body of the robot would be useful so the robot can plan/lookahead far in advance using perceptual data like depth or lidar. In many cases, we found the level sets of the SDF were only accurate a few centimeters from the surface. Or, we could get more accurate off-surface distances at the expense of on-surface resolution, which was also undesirable. This is likely a matter of scaling. For example, if the SDF is off by a cm or two on-surface it is very noticible, but at a distance of a meter or so from the robot this becomes negligible. 


To fix this, we trained the model in two stages. The first stage learns the on-surface model, $\phi$, with high-resolution. Then, the off-surface distance are added in using a secondary, blending neural network $\sigma$ that exploits the fact that all distances far from the robot are approximately equal to the distance from the center of the robot $d_{\text{nom}}$. 


$d=\phi(x, y, z, q) + \sigma(x, y, z, q) d_{\text{nom}}$

Here's an example of some level sets near the surface of the body using this two stage blending strategy:


https://github.com/user-attachments/assets/d1153575-e9dc-4173-bc63-c0e7a1c7caba


Here are some level sets far from the surface of the body with the same model:



https://github.com/user-attachments/assets/d60c6111-7a83-416c-9138-111d3b566035



Signed distance functions, although useful for planning, collision detection etc., only provide geometric information which can be limiting. For instance, let's say we want to touch an object with a specific part of our body, an SDF only tells us how close we are, and the direction normal to the surface. I was inspired by correspondence models like [dense object descriptors for manipulation](https://arxiv.org/abs/1806.08756) to create a similar model for a quadruped, that we could use to differentiate all parts of the body in a continuous yet unified way. 

I called these _dense self-descriptors_, which are defined as $\gamma(x, y, z, q) \in \mathbb{R}^3$. Where spatial inputs lie on the surface of the robot and are mapped to template coordinates $(x_{\text{T}}, y_{\text{T}}, z_{\text{T}})$ that correspond to that same point on the robot in a nominal or template configuration $q_T$. 

Pictured below are these dense self-descriptors for the robot in the template configuration (i.e., $(x_{\text{T}}, y_{\text{T}}, z_{\text{T}})=(x, y, z)$ ) overlayed on the predicted SDF:


<img width="528" height="400" alt="image" src="https://github.com/user-attachments/assets/22556214-8799-48af-b488-be47d8a71abd" />

If we modify the configuration of the robot, the continuous descriptors for a given body part remain consistent:

<img width="805" height="500" alt="image" src="https://github.com/user-attachments/assets/fe63b1ba-f318-40e9-b47b-7d6343f22df5" />

We also learn an inverse model $\gamma^{-1}(x_{\text{T}}, y_{\text{T}}, z_{\text{T}} q) \in \mathbb{R}^3$ that enables us to track a descriptor as the configuration changes. Pictured below are outputs of the inverse or tracking model, given a batch of template points and a random joint configuration.

<img width="620" height="500" alt="image" src="https://github.com/user-attachments/assets/26e65a1a-2225-4f61-b7bb-3c0de7d5bf6d" />

Note that this is not a rendered SDF but points from the template SDF after being passed through the tracking/inverse model with the different configuration with color map denoting their template or pretransformed spatial coordinates.

These desriptors could be used as inputs to a learned dynamics model or policy. In the end we imagined the pipeline would sample key points from a scene, whether modeled via meshes, SDFs, or just from depth/lidar then transform them through the self-model/SDF. Then, the SDF can be used to project these key points on to the body, giving the closet point of the robot to the key points in the environment. Then, these projected on-surface points can be used to obtain their corresponding descriptors. Key points on the robot can be also tracked simultaneously via the inverse model.

In the end, this allows the robot to answer the questions:

1. _How close am I_ to a given point in the environment/scene?
2. _What part of me is closest_ to a given point in the the environment/scene?
3. _Where is my ****_ and how close is it to ****?

It is typical, in classical legged robots, to mainly consider contacts with the feet. However, allowing the robot to plan with these questions in mind may unlock better whole-body manipulation/pedipulation. I imagine tasks like a humanoid opening a door with shoulder or hips like a human, or pushing in a chair with its thigh after standing up, or carrying large objects over the shoulder. 
