from .Base import *
from .utils import get_param


class Embeddings(nn.Module):            
    
    def __init__(self, num_ent, num_rel, params):
        super(Embeddings, self).__init__()
        self.encoder_name = params.DECODER.NAME.lower()
        self.embed_dim = params.ENCODER.EMBED_DIM

        self.init_embed = get_param((num_ent, params.ENCODER.EMBED_DIM))
        self.num_ent = num_ent
        
        if params.ENCODER.NUM_BASES > 0:
            self.init_rel = get_param((num_rel, params.ENCODER.EMBED_DIM))
            self.num_rel = num_rel
        else:
            if self.encoder_name == 'transe':
                self.init_rel = get_param((num_rel, params.ENCODER.EMBED_DIM))
            else:
                self.init_rel = get_param((num_rel * 2, params.ENCODER.EMBED_DIM))
            self.num_rel = num_rel * 2
    
    def forward(self):                                                  
        
        ent_embeddings = self.init_embed
        rel_embeddings = self.init_rel
        if self.encoder_name == 'transe':
                rel_embeddings = torch.cat([self.init_rel, -self.init_rel], dim=0)
        
        return ent_embeddings, rel_embeddings
    
    def __repr__(self):                 
        msg = ''
        
        return 'Embeddings(\n'\
            '  (ent_embeddings): Parameters({}, {}),\n'\
            '  (rel_embeddings): Parameters({}, {})\n'\
            '  )'.format(self.num_ent, self.embed_dim, self.num_rel, self.embed_dim, msg)