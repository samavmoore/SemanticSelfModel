# Semantic Self-Modeling for Whole-body Awareness and Planning

We started this projected with the intention of extending Boyuan Chen's work _Full-body visual self-modeling of robot morphologies_ in Science Robotics ([Arxiv](https://arxiv.org/abs/2111.06389)) to legged robots. The original framework uses implicit 3-D representations (DeepSDFs with siren layers) conditioned on joint-angles to model robot bodies from pointcloud observations. Then, they use this model (shown below) for a range of downstream planning tasks.

<img width="858" height="252" alt="VSM_science_robotics" src="https://github.com/user-attachments/assets/43cd4175-4c57-4177-b07c-a07c2f9796e7" />

We noticed issues with scalability when attempting to apply the original archtitecture directly to a quadruped. So, the one thing the code here does is test all of the architectures in the figure below on the unitree Go2:

<img width="654" height="600" alt="VSM_architectures" src="https://github.com/user-attachments/assets/920c7898-0116-4782-ba14-5a3d27bdefdc" />

The blue layers use siren activations while gray use ReLU. The Hadamard product indicates the product of features, almost like attention x_l+1 = sin(w * z_q^l * x_l + b). The models that work the best (b and e), split the leg data, q = [q1, q2, q3, q4], into seperate encoders before fusing into a global representation. 

Here's an example rendering of the self-model/joint-conditioned SDF (using (b)) in a random configuration at test time:

https://github.com/user-attachments/assets/38ab3e2f-2121-4dac-9e59-a5e46c6570ee


The way that these models are typically trained, though, can make capturing accurate off-surface distances very challenging. Accurate signed distances far from the body of the robot would be useful so the robot can plan or lookahead far in advance using perceptual data like depth or lidar. In many cases, we found the level set of the SDF were only accurate a few centimeters from the surface. Or, we could get more accurate off-surface distances at the expense of on-surface resolution, which was also undesirable. This is likely a matter of scaling. For example, if the SDF is off by a cm or two on-surface this is very noticible, but at a distance of a meter or so this becomes negligible. To fix this, we trained the model in two stages. The first stage learned the on-surface model with high-resolution. Then, the off-surface distance are added in using a secondary, blending neural network that exploits the fact that all distances far from the robot are approximately equal to the distance from the center of the robot. I got this idea from another work, but the exact paper escapes me as I write this. 

Here's an example of level sets near the surface of the body using this two stage blending strategy:


https://github.com/user-attachments/assets/d1153575-e9dc-4173-bc63-c0e7a1c7caba


Here are the level sets far from the surface of the body with the same model:



https://github.com/user-attachments/assets/d60c6111-7a83-416c-9138-111d3b566035



Signed distance functions, although useful for planning, only provide the robots geometry which can be limiting. For instance, lets say we want to touch an object with a specific part of our body, an SDF only tells us how close we are, and the direction normal to the surface. To combat t

