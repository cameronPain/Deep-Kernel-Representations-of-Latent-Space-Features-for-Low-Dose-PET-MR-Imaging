# Deep-Kernel-Representations-of-Latent-Space-Features-for-Low-Dose-PET-MR-Imaging
Citation:
Pain et al., 2024, "Deep kernel representations of latent space
features for low-dose PET-MR imaging robust to
variable dose reduction", 

A deep learning-based method for low-dose MR-PET image processing robust to previously unseen dose reduction factors.
We represent latent-space features with kernel functions derived from MR features to provide a pre-defined level of anatomical guidance during training.
Additionally, we apply information constraints to deep latent-space features during training to further improve performance on previously unseen dose-reduction factors.

###
### Implementation
###

We implement our method using Tensorflow version 2.12.
Our network is a Unet shape with 5 layers. Each layer of the encoder uses layer-specific kernel functions to encode features.
A single decoder is used to map MR features and kernel features to the target domain.

###
### Training
###

Our data loader takes input  low-dose PET and MR images of dimensions (150,192,192,3) (axial,anterior-posterior,left-right, ld PT T1 nd T2) in the .npy file format.
Target data is standard-dose PET of dimensions (150,192,192,1) in the .npy file format.
These dimensions correspond to 1mm^3 isotropic voxels.

We define the hyperparameters for the information constraints gamma_{min} and gamma_{max} which are useful for preventing bias in very low-dose images (> x500 low-dose in our experiments).
Generally we found a range gamma \in (0.1,0.0005) to be useful for x500 and x1000 dose reduction.

Kernel patch-size provides the strongest regularisation for previously unseen PET. 
For in-distribution a network trained with 1x1 patch size reverts to a conventional approach with no explicit MR guidance.
For large dose reduction factors (>x500) we achieve good regularisation using a patch size of >( h^{l}/3, w^{l}/3)
where h^{l} and w^{l} are the feature map height and width for each layer l.

The stride factor is used to reduce the total number of FLOPs required for a single forward pass at the expense of some modelling capacity. 
This can be chosen based on hardware and time limitations.

###
### Evaluation
###
We have provided a collection of pre-trained weights trained at varying values of patch-size, stride factor and information constraint for testing.
We have provided an example test case also.

Data Link: https://monashuni-my.sharepoint.com/:u:/g/personal/cameron_pain_monash_edu/Ed3CeYoCAfZBtzx_w8crLGwB6Yu-B5lopHCizJymBXyX1w?e=pUVIuC

Weights Link: https://monashuni-my.sharepoint.com/:u:/g/personal/cameron_pain_monash_edu/EclbWBse1lNCnky2Vvc3r5MBfqBUMExg9IWI0UGKamWWJQ?e=CgXG1f


