from .Base import *
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class Test(nn.Module):
    def __init__(self, n):
        super(Test, self).__init__()
        self.n = n
        
    def forward(self, x):
        print('Num:', self.n, x.size())
        return x


# TransE Class
class TransE(BaseModel):
    
    def forward(self, sub_embed, rel_embed, all_embed):
        
        obj_embed = sub_embed + rel_embed
        x = self.p.DECODER.GAMMA - torch.norm(obj_embed.unsqueeze(1) - all_embed, p=1, dim=2)
        
        return torch.sigmoid(x)


# DistMult Class
class DistMult(BaseModel):
    
    def forward(self, sub_embed, rel_embed, all_embed):
        obj_embed = sub_embed * rel_embed
        x = torch.mm(obj_embed, all_embed.t())
        # x += self.bias.expand_as(x)
        
        return torch.sigmoid(x)

# ConvE Class
class ConvE(BaseModel):             
    
    def __init__(self, num_ent, params, device):
        super(ConvE, self).__init__(params, device)

        if isinstance(params.DECODER.KER_SZ, str):
            ker_size_list = [int(x) for x in params.DECODER.KER_SZ.split(',')]
        else: 
            ker_size_list = [params.DECODER.KER_SZ]
        self.num = len(ker_size_list)
        _tmp_num_split = params.DECODER.NUM_FILTERS // self.num
        steps = None
        if params.DECODER.NUM_FILTERS % self.num == 0:
            steps = [_tmp_num_split] * self.num
        else:
            steps = [params.DECODER.NUM_FILTERS // (self.num - 1)] * (self.num -1) + [params.DECODER.NUM_FILTERS % (self.num - 1)]
        
        _step = _tmp_num_split if _tmp_num_split * self.num == params.DECODER.NUM_FILTERS else _tmp_num_split + 1
        
        self.convolutions = nn.ModuleList()
        
        for i, k in enumerate(steps):
            i_k_sz = ker_size_list[i]
            
            flat_sz_h = int(2 * params.DECODER.K_W) - i_k_sz + 1
            flat_sz_w = self.p.DECODER.K_H - i_k_sz + 1
            flat_sz = flat_sz_h * flat_sz_w * k
            
            self.convolutions.append(nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Conv2d(1, out_channels=k, kernel_size=(i_k_sz, i_k_sz), stride=1, padding=0, bias=params.DECODER.CONVE_BIAS),
                nn.BatchNorm2d(k),
                nn.ReLU(),
                nn.Dropout(params.DECODER.DECODER_FEAT_DROP),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(flat_sz, params.ENCODER.GCN_OUT_CHANNELS),
                nn.Dropout(params.DECODER.DECODER_HID_DROP),
                nn.BatchNorm1d(num_features=params.ENCODER.GCN_OUT_CHANNELS),
                nn.ReLU(),
            ))

        self.register_parameter('bias', nn.Parameter(torch.zeros(num_ent), requires_grad=True))
    
    def concat(self, ent_embed, rel_embed):             
        ent_embed = ent_embed.view(-1, 1, self.p.ENCODER.GCN_OUT_CHANNELS)
        rel_embed = rel_embed.view(-1, 1, self.p.ENCODER.GCN_OUT_CHANNELS)
        stack_inp = torch.cat((ent_embed, rel_embed), dim=1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.DECODER.K_W, self.p.DECODER.K_H))
        return stack_inp
    
    def forward(self, sub_embed, rel_embed, all_embed):                 
        # print(sub_embed)
        stk_inp = self.concat(sub_embed, rel_embed)
        # print(sub_embed.size(), stk_inp.size())
        
        pred_lists = []
        
        for i in range(self.num):
            x = self.convolutions[i](stk_inp)
            x = torch.mm(x, all_embed.transpose(1, 0))
            pred_lists.append(x)
        
        pred = pred_lists[0]
        
        for i in range(1, self.num):
            pred += pred_lists[i]
        
        pred += self.bias.expand_as(pred)
        score = torch.sigmoid(pred)
        return score


