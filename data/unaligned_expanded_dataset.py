import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch


class UnalignedExpandedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    
    
    Only changes is that order of images in A is permuted, and images from train_expanded are added
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, 'trainA')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, 'trainB')  # create a path '/path/to/data/trainB'
        # #self.dir_expanded_A = os.path.join(opt.dataroot, 'train_expandedA')  # create a path '/path/to/data/train_expandedA'
        # #self.dir_expanded_B = os.path.join(opt.dataroot, 'train_expandedB')  # create a path '/path/to/data/train_expandedB'

        # self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        # self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        # #self.A_expanded_paths = sorted(make_dataset(self.dir_expanded_A, opt.max_dataset_size))   # load images from '/path/to/data/train_expandedA'
        # #self.B_expanded_paths = sorted(make_dataset(self.dir_expanded_B, opt.max_dataset_size))    # load images from '/path/to/data/train_expandedB'
        
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # #self.A_random_permutation = np.random.permutation(self.A_size) # Permute the order of images in A !!!
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = transforms.Compose([get_transform(self.opt,convert=False),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
        self.transform_B = transforms.Compose([get_transform(self.opt,convert=False),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # #A_path = self.A_paths[self.A_random_permutation[index % self.A_size]]  # make sure index is within then range
        # #A_expanded_path = self.A_expanded_paths[self.A_random_permutation[index % self.A_size]]
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # #A_expanded_path = self.A_expanded_paths[index % self.A_size]
        # if self.opt.serial_batches:   # make sure index is within then range
            # index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
            # index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        # B_expanded_path = self.B_expanded_paths[index_B]
        # A_img = Image.open(A_path)#.convert('RGBA')
        # B_img = Image.open(B_path)#.convert('RGBA')
        # #A_expanded_img = Image.open(A_expanded_path).convert('LA')
        # #B_expanded_img = Image.open(B_expanded_path).convert('LA')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)
        # #A_expanded = torch.cat((A,self.transform_expanded_A(A_expanded_img)))
        # #B_expanded = torch.cat((B,self.transform_expanded_B(B_expanded_img)))
        
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)#.convert('RGB')
        B_img = Image.open(B_path)#.convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
