import os
import numpy as np

import torch
import torch.utils.data as data
from spr_load import load_vin


class SprData(data.Dataset):
    def __init__(self, datadir, imsize, mode='local',
                 annotation='human', train=True, scale=50,
                 load_instructions=False,
                 transform=None, target_transform=None):
        assert os.path.isdir(datadir) # Must be .npz format
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set
        num_train = 1566 #local maps
        num_test = 399
        self.mode = mode
        self.annotation = annotation
        self.datadir = datadir
        self.load_instructions = load_instructions
        # self.images, self.S1, self.S2, self.labels =  \
        #                         self._process(file, self.train)
        self.dataset, self.values, self.text_vocab = load_vin(self.datadir, self.mode, self.annotation, num_train,
                                num_test, train=self.train, load_instructions=self.load_instructions)
        self.text_vocab_size   = len(self.text_vocab) + 1
        data_type = 'train' if self.train else 'test'
        print "Loaded %s %s examples" %(len(self.dataset), data_type)

    def __getitem__(self, index):
        # img = self.images[index]
        # s1 = self.S1[index]
        # s2 = self.S2[index]
        # label = self.labels[index]
        img, s1, s2, label = self.dataset[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        # else: # Internal default transform: Just to Tensor
        #     img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.load_instructions:
            return img, int(s1), int(s2), int(label)
        else:
            return img, int(s1), int(s2), int(label)
        

    def __len__(self):
        return len(self.dataset)
        

    # def _process(self, file, train):
    #     """Data format: A list, [train data, test data]
    #     Each data sample: label, S1, S2, Images, in this order.
    #     """
    #     with np.load(file) as f:
    #         if train:
    #             images = f['arr_0']
    #             S1 = f['arr_1']
    #             S2 = f['arr_2']
    #             labels = f['arr_3']
    #         else:
    #             images = f['arr_4']
    #             S1 = f['arr_5']
    #             S2 = f['arr_6']
    #             labels = f['arr_7']
    #     # Set proper datatypes
    #     images = images.astype(np.float32)
    #     S1 = S1.astype(int) # (S1, S2) location are integers
    #     S2 = S2.astype(int)
    #     labels = labels.astype(int) # labels are integers
    #     # Print number of samples
    #     if train:
    #         print("Number of Train Samples: {0}".format(images.shape[0]))
    #     else:
    #         print("Number of Test Samples: {0}".format(images.shape[0]))
    #     return images, S1, S2, labels
