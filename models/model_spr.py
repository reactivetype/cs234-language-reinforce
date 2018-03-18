import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter

#Models
from lookup_model import LookupModel
from text_model import TextModel
from attention_heatmap import AttentionHeatmap

class SprVin(nn.Module): 
    def __init__(self, config, heatmap_model, object_model):
        super(SprVin, self).__init__()
        self.config = config
        self.heatmap_model = heatmap_model
        self.object_model = object_model
        self.h = nn.Conv2d(in_channels=config.l_i, 
                           out_channels=config.l_h, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=True)
        self.r = nn.Conv2d(in_channels=config.l_h, 
                           out_channels=1, 
                           kernel_size=(1, 1), 
                           stride=1, padding=0,
                           bias=False)
        self.q = nn.Conv2d(in_channels=1, 
                           out_channels=config.l_q, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=False)
        self.fc = nn.Linear(in_features=config.l_q, 
                            out_features=4,
                            bias=False)
        self.w = Parameter(torch.zeros(config.l_q,1,3,3), requires_grad=True)
        self.sm = nn.Softmax()


    def forward(self, layout, obj_map, inst, S1, S2, config):
        obj_embed = self.object_model.forward(obj_map)
        heatmap = self.heatmap_model((obj_embed, inst))
        X = torch.cat([layout, heatmap], dim=1)
        h = self.h(X)

        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(torch.cat([r, v], 1), 
                         torch.cat([self.q.weight, self.w], 1),
                         stride=1, 
                         padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = F.conv2d(torch.cat([r, v], 1), 
                     torch.cat([self.q.weight, self.w], 1),
                     stride=1, 
                     padding=1)

        slice_s1 = S1.long().expand(config.imsize, 1, config.l_q, q.size(0))
        slice_s1 = slice_s1.permute(3, 2, 1, 0)
        q_out = q.gather(2, slice_s1).squeeze(2)

        slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        slice_s2 = slice_s2.permute(2, 1, 0)
        q_out = q_out.gather(2, slice_s2).squeeze(2)

        logits = self.fc(q_out)
        return logits, self.sm(logits), v, r, heatmap

if __name__ == '__main__':
    #create object embedding model
    objects = 10
    obj_embed = 7
    object_model = LookupModel(objects, embed_dim=obj_embed)

    #text lstm model
    attn_kernel = 3
    attn_in = obj_embed
    attn_out = 1 # no. of heatmap channels
    lstm_out = (attn_kernel**2) * attn_in * attn_out
    vocab_size = 300
    instruction_model = TextModel(vocab_size, ninp=15,
                                nhid=30, nlayers=1,
                                out_dim=lstm_out)
    heatmap_model = AttentionHeatmap(instruction_model,
                                    attn_kernel,
                                    attn_in,
                                    attn_out)

