# Author        : Kaiyuan Wang
# Affiliation   : ARCLab @ UC San Diego
# Email         : k5wang@ucsd.edu
# Created on    : Mon Aug 21 2023


'''Residential Robot Demonstration Dataset (R2D2) data loader. '''

import sys, torch, glob, itertools
import numpy as np
import torchvision.transforms as transforms
from os.path import join, abspath, expanduser, isdir, exists, dirname
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from tqdm import tqdm

BASE_DIR = abspath(join(dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
from r2d2.trajectory_utils.misc import load_trajectory

INTRINSICS_TABLE = {
    '23404442_left': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
    '23404442_right': np.array([[522.5022583, 0. , 640.49182129], [0., 522.5022583, 353.23074341],[0.,0.,1.]]),
    '29838012_left': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
    '29838012_right': np.array([[523.22283936, 0. , 639.23681641], [0., 523.22283936, 352.50140381],[0.,0.,1.]]),
    '19824535_left': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]]),
    '19824535_right': np.array([[697.90771484, 0. , 649.59765625], [0., 697.90771484, 354.90002441],[0.,0.,1.]])
}

DIST_COEFFS_TABLE = {
    '23404442' : np.array([0., 0., 0., 0., 0.]),
    '29838012' : np.array([0., 0., 0., 0., 0.]),
    '19824535' : np.array([0., 0., 0., 0., 0.])
}

PANDA_MESH_FILES = [BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link0/link0.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link1/link1.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link2/link2.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link3/link3.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link4/link4.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link5/link5.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link6/link6.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/link7/link7.obj",
                BASE_DIR + "/ctrnet/urdfs/Panda/meshes/visual/hand/hand.obj"]


class R2D2DatasetBlock(Dataset):

    def __init__(self, data_folder='~/Desktop/data/r2d2_household/pen_out_several/Fri_Apr_21_10_35_02_2023',
        camera_id='23404442_left', scale=(1.,1.), trans_to_tensor=None,
    ):
        ''' A block of R2D2 dataset. Concat all R2D2DatasetBlocks to construct full R2D2 dataset.
        This class can be used as a stand-alone dataset by itself.

        Inputs:
            - data_folder:      path to data dir, e.g. '~/data/pen_in/Fri_Apr_21_10_42_22_*'
            - scale: tuple      H, W scale factor for image size
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
        self.camera_id = camera_id

        self.data_folder = abspath(expanduser(data_folder))
        self.h5_path = join(self.data_folder, "trajectory.h5")
        self.mp4_path = join(self.data_folder, "recordings/MP4")
        self.timestep_list = load_trajectory(filepath=self.h5_path, recording_folderpath=self.mp4_path)

        # list of (time, camera_id) pairs
        # self.data_ids = list(itertools.product(range(len(self.timestep_list)), self.camera_ids))

        img_sample = self.timestep_list[0]['observation']['image'][self.camera_id]
        self.height = int(img_sample.shape[0] * self.scale[0])
        self.width = int(img_sample.shape[1] * self.scale[1])

    def __len__(self):  return len(self.timestep_list)

    def __getitem__(self, idx):

        # (time, camera_id) = self.data_ids[idx]
        img = self.timestep_list[idx]['observation']['image'][self.camera_id] # dict: camera_id -> img
        img = Image.fromarray(img) # (H,W,3) uint8
        img = img.resize((self.width, self.height))
        img = self.trans_to_tensor(img)                 # (3,H,W)
        joint_angle = self.timestep_list[idx]['action']['joint_position']
        joint_angle = torch.tensor(joint_angle, dtype=torch.float)

        return img, joint_angle

def get_camera_intrinsics(camera_id):
    ''' Get camera (3, 3) intrinsics for R2D2 dataset.'''

    assert camera_id in ['23404442_left', '23404442_right', '29838012_left','29838012_right', '19824535_left', '19824535_right'], \
        f"camera_id [{camera_id}] not recognized. Please choose from ['23404442_left', '23404442_right', '29838012_left','29838012_right', '19824535_left', '19824535_right']"

    return INTRINSICS_TABLE[camera_id]

def get_robot_mesh_files():
    ''' Get PANDA robot mesh files for R2D2 dataset.'''
    return PANDA_MESH_FILES

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

    print(" Executing test run of r2d2_data.py ... \n")

    data_dir ='~/Desktop/data/r2d2_household/pen_out_several'
    if not exists(expanduser(data_dir)):
        raise ValueError(f"Data directory [{data_dir}] does not exist. Please change file path in main function of r2d2_data.py")
    dataset = get_r2d2_dataset(data_dir, max_num_blocks=1)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"Dataset length: [{len(dataset)}]\n")
    for i, batch in (pbar1:=tqdm(enumerate(data_loader), desc="Enumerating dataset")):
        img, joint_angle = batch
        pbar1.set_description(f"Loading batch {i} out of {len(data_loader)}")
        pbar1.set_postfix({"img_shape": img.shape, "joint_angle_shape": joint_angle.shape})
    pbar1.close()