import os
import numpy as np

import torch
import torch.utils.data as data
from spr_load import load_vin, load_vin_instructions


class SprTerminal(data.Dataset):
    def __init__(self, datadir, imsize, mode='local',
                 annotation='human', train=True, scale=50,
                 load_instructions=False, transform=None,
                 target_transform=None):
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
        self.dataset, self.values, self.text_vocab = load_vin(self.datadir, self.mode, self.annotation,
                                                              num_train, num_test, train=self.train,
                                                              load_instructions=self.load_instructions)
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
        
class SprInstructions(data.Dataset):
    def __init__(self, datadir, mode='local',
                 annotation='human', train=True, scale=50,
                 transform=None, target_transform=None,
                 actions=4, with_rewards=True):
        assert os.path.isdir(datadir) # Must be .npz format
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set
        self.num_train = 1566 #local maps
        self.num_test = 399
        self.mode = mode
        self.annotation = annotation
        self.datadir = datadir
        self.scale = scale
        self.actions = actions
        self.with_rewards = with_rewards
        self.load_data()
        self.text_vocab_size   = len(self.text_vocab) + 1
        data_type = 'train' if self.train else 'test'
        print "Loaded %s %s examples" %(len(self.layouts), data_type)

    def load_data(self):
        self.layouts, self.objects, self.inst_indices,\
        self.s1, self.s2, self.goals, self.labels, self.values,\
        self.text_vocab, self.object_vocab_size,\
        self.rewards, self.instructions = load_vin_instructions(self.datadir, self.mode, self.annotation,
                                                                  self.num_train, self.num_test, train=self.train,
                                                                  scale=self.scale, actions=self.actions,
                                                                  with_rewards=self.with_rewards)
        print self.goals[0], len(self.goals)
        return

    def __getitem__(self, index):
        layout = np.array(self.layouts[index], dtype=np.float32)
        object_map = np.array(self.objects[index], dtype=int)
        inst = np.array(self.inst_indices[index], dtype=int)
        s1 = int(self.s1[index])
        s2 = int(self.s2[index])
        label = int(self.labels[index])
        goal = self.goals[index]


        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return layout, object_map, inst, s1, s2, goal, label
        

    def __len__(self):
        return len(self.layouts)
        
if __name__ == "__main__":
    data_dir = "spr_data/"
    spr_data = SprInstructions(data_dir)
    dataloader = torch.utils.data.DataLoader(spr_data, batch_size=128, shuffle=True)
    from IPython import embed
    embed()