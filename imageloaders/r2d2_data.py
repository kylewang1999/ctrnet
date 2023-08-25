# Author        : Kaiyuan Wang
# Affiliation   : ARCLab @ UC San Diego
# Email         : k5wang@ucsd.edu
# Created on    : Mon Aug 21 2023

'''Residential Robot Demonstration Dataset (R2D2) data loader. '''

import os, sys, torch, glob, itertools, time
import torchvision.transforms as transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from os.path import join, abspath, expanduser, isdir
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from tqdm import tqdm

from r2d2.trajectory_utils.misc import load_trajectory


''' The following are the camera intrinsics for R2D2 dataset.
SN:  23404442 {'23404442_left': {'cameraMatrix': array([[522.5022583 ,   0.        , 640.49182129],
       [  0.        , 522.5022583 , 353.23074341],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}, '23404442_right': {'cameraMatrix': array([[522.5022583 ,   0.        , 640.49182129],
       [  0.        , 522.5022583 , 353.23074341],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}}
SN:  29838012 {'29838012_left': {'cameraMatrix': array([[523.22283936,   0.        , 639.23681641],
       [  0.        , 523.22283936, 352.50140381],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}, '29838012_right': {'cameraMatrix': array([[523.22283936,   0.        , 639.23681641],
       [  0.        , 523.22283936, 352.50140381],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}}
SN:  19824535 {'19824535_left': {'cameraMatrix': array([[697.90771484,   0.        , 649.59765625],
       [  0.        , 697.90771484, 354.90002441],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}, '19824535_right': {'cameraMatrix': array([[697.90771484,   0.        , 649.59765625],
       [  0.        , 697.90771484, 354.90002441],
       [  0.        ,   0.        ,   1.        ]]), 'distCoeffs': array([0., 0., 0., 0., 0.])}}
'''


class R2D2DatasetBlock(Dataset):

    def __init__(self, data_folder='~/Desktop/data/r2d2_household/pen_out_several/Fri_Apr_21_10_35_02_2023', 
        camera_ids=['23404442_left', '23404442_right', '29838012_left','29838012_right'],
        scale=(2/3, 1/2), trans_to_tensor=None,
    ):
        ''' A block of R2D2 dataset. Concat all R2D2DatasetBlocks to construct full R2D2 dataset.
        This class can be used as a stand-alone dataset by itself.

        Inputs:
            - data_folder:      path to data dir, e.g. '~/data/pen_in/Fri_Apr_21_10_42_22_*'
            - scale: tuple      W, H scale factor for image size 
            - trans_to_tensor:  torchvision.transforms.Compose object
            - camera_id:        a list of camera_id's selected from
                                {23404442_left', '23404442_right', '29838012_left','29838012_right'}
              NOTE: '19824535_left', '19824535_right are excluded because they locate on the gripper
        '''

        if trans_to_tensor is None:
            self.trans_to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: self.trans_to_tensor = trans_to_tensor
        
        self.scale = scale
        self.camera_ids = camera_ids

        self.data_folder = abspath(expanduser(data_folder))
        self.h5_path = join(self.data_folder, "trajectory.h5")
        self.mp4_path = join(self.data_folder, "recordings/MP4")
        self.timestep_list = load_trajectory(filepath=self.h5_path, recording_folderpath=self.mp4_path)

        # list of (time, camera_id) pairs
        self.data_ids = list(itertools.product(range(len(self.timestep_list)), self.camera_ids)) 

        img_sample = self.timestep_list[0]['observation']['image'][self.camera_ids[0]]
        self.width = int(img_sample.shape[0] * self.scale[0])
        self.height = int(img_sample.shape[1] * self.scale[1])

    def __len__(self):  return len(self.data_ids)

    def __getitem__(self, idx):

        (time, camera_id) = self.data_ids[idx]
        
        img = self.timestep_list[time]['observation']['image'][camera_id] # dict: camera_id -> img 
        img = Image.fromarray(img) # (H,W,3) uint8 
        img = img.resize((self.width, self.height)) 
        img = self.trans_to_tensor(img)                 # (3,H,W)
        joint_angle = self.timestep_list[time]['action']['joint_position']
        joint_angle = torch.tensor(joint_angle, dtype=torch.float)

        return img, joint_angle



def get_r2d2_dataset(data_folder='~/Desktop/data/r2d2_household/pen_out_several', 
    camera_ids=['23404442_left', '23404442_right', '29838012_left','29838012_right'],
    max_num_blocks=2, scale=(2/3, 1/2), trans_to_tensor=None
):
    '''
    Inputs:
        ...
        - max_num_blocks: maximum number of blocks (i.e. number of dirs) to load.
    '''

    paths_to_blocks = sorted([path for path \
        in glob.glob(join(abspath(expanduser(data_folder)), '*')) if isdir(path)])

    max_num_blocks = min(max_num_blocks, len(paths_to_blocks))
    paths_to_blocks = paths_to_blocks[:max_num_blocks]

    print(f'\nConstructing dataset blocks. This might take a while...\n')
    dataset_blocks = []
    for path in (pbar:=tqdm(paths_to_blocks, desc="Constructing dataset blocks")):
        dataset_blocks.append(R2D2DatasetBlock(
            path, camera_ids=camera_ids, scale=scale,trans_to_tensor=trans_to_tensor))
        pbar.set_postfix({"block length": len(dataset_blocks[-1])})
    pbar.close()


    return ConcatDataset(dataset_blocks)


    
if __name__ == "__main__":

    data_dir ='~/Desktop/data/r2d2_household/pen_out_several'
    dataset = get_r2d2_dataset(data_dir, max_num_blocks=1)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Dataset length: [{len(dataset)}]\n")
    for i, batch in (pbar1:=tqdm(enumerate(data_loader), desc="Enumerating dataset")):
        img, joint_angle = batch
        pbar1.set_description(f"Loading batch {i} out of {len(data_loader)}")
        pbar1.set_postfix({"img_shape": img.shape, "joint_angle_shape": joint_angle.shape})
    pbar1.close()