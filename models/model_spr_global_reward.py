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

class SprVinReward(nn.Module): 
    def __init__(self, config, heatmap_model, object_model, actions=4):
        super(SprVinReward, self).__init__()
        self.config = config
        self.heatmap_model = heatmap_model
        self.object_model = object_model
        self.actions = actions
        self.h0 = nn.Conv2d(in_channels=config.l_i, 
                           out_channels=config.l_h/4, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=True)
        self.h1 = nn.Conv2d(in_channels=config.l_h/4, 
                           out_channels=config.l_h/2, 
                           kernel_size=(3, 3), 
                           stride=1, padding=1,
                           bias=True)
        self.h2 = nn.Conv2d(in_channels=config.l_h/2, 
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
                            out_features=self.actions,
                            bias=False)
        self.w = Parameter(torch.zeros(config.l_q,1,3,3), requires_grad=True)
        self.sm = nn.Softmax()
        self.positions = Variable( self.__init_positions(config.imsize).cuda() )
        self.positions_batch = self.positions.repeat(config.batch_size,1,1,1)
        self.drop = True
        self.conv1_drop = nn.Dropout2d(p=0.2)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.conv3_drop = nn.Dropout2d(p=0.2)
        return

    def __init_positions(self, map_dim):
        row = torch.arange(0,map_dim).unsqueeze(1).repeat(1,map_dim)
        col = torch.arange(0,map_dim).repeat(map_dim,1)
        positions = torch.stack( (row, col) )
        return positions

    def _global(self, global_coeffs, map_dim=10):
        pos_coeffs = global_coeffs[:,:-1]
        bias = global_coeffs[:,-1]

        coeffs_batch = pos_coeffs.unsqueeze(-1).unsqueeze(-1).repeat(1,1,map_dim,map_dim)
        bias_batch = bias.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,map_dim,map_dim)
        
        ## sum over row, col and add bias
        obj_global = (coeffs_batch * self.positions_batch).sum(1, keepdim=True) + bias_batch
        return obj_global

    def forward(self, layout, obj_map, inst, S1, S2, config):
        obj_embed = self.object_model.forward(obj_map)
        local_heatmap, global_coeffs = self.heatmap_model((obj_embed, inst))
        global_heatmap = self._global(global_coeffs)
        X = torch.cat([layout, local_heatmap, global_heatmap], dim=1)
        h0 = self.h0(X)
        if self.drop:
            h0 = F.relu(self.conv1_drop(h0))
        
        h1 = self.h1(h0)
        if self.drop:
            h1 = F.relu(self.conv2_drop(h1))

        h2 = self.h2(h1)
        if self.drop:
            h2 = self.conv3_drop(h2)

        r = self.r(h2)
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
        return logits, self.sm(logits), v, r, local_heatmap

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

