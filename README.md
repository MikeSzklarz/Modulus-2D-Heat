# 2D Fixed Heat Transfer using Modulus
![heat_transfer_animation](https://github.com/user-attachments/assets/2fd5f151-9aff-4adc-9263-9b7f97fec2d3)

## Required
- Ubuntu 20.04 or Linux 5.13 kernel
- Git
- Conda
  
### Cloning repo
`git clone https://github.com/MikeSzklarz/Modulus-2D-Heat.git`  
`cd Modulus-2D-Heat/`

### Create Conda Environment
`conda env create -f environment.yml`  
`conda activate modulus_test`

### Train Model
`python train_2d_bound.py`

### Visualize using TensorBoard
`tensorboard --logdir outputs`  
Current implementation works locally for visualizing results  
