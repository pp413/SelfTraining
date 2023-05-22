from .Base import *
from .embeddings import Embeddings
from .decoder import TransE, DistMult, ConvE



__all__ = ['Embed_TransE', 'Embed_DistMult', 'Embed_ConvE']
         # 'Embed_InteractE' 'CompGCN_TransE', 'CompGCN_DistMult', 'CompGCN_InteractE', 'CompGCN_ConvE'


# ############################################################################################
#                 Models
# ############################################################################################

class Model(BaseModel):                 
    
    def __init__(self, params, num_ent, num_rel, edge_index, edge_type, device):
        super(Model, self).__init__(params, device)
        
        self.edge_index = edge_index.long()
        self.edge_type = edge_type.long()
        self.num_ent = num_ent
        self.num_rel = num_rel
        
        self.create_embeddings()

        self.create_encoder()
        self.create_decoder()
        # self.create_decoder()
        
        self.bce_loss = nn.BCELoss()
        
        dim = self.p.ENCODER.EMBED_DIM if self.encoder is None else self.p.ENCODER.GCN_OUT_CHANNELS
        
        l = params.ENCODER.GCN_NUM_LAYERS if params.ENCODER.NAME != 'Embed' else 0
        
        self.final_ent_embeddings = Parameter(torch.empty(self.num_ent, dim, device=device), requires_grad=False)
        self.final_rel_embeddings = Parameter(torch.empty(2 * self.num_rel + l, dim, device=device), requires_grad=False)
    
    def loss(self, pred, label):                                
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)
        label = label.to(self.device)
        
        if isinstance(pred, tuple):
            return self.bce_loss(pred[0], label) + self.bce_loss(pred[2], label) + self.bce_loss(pred[2], label)
        else:
            return self.bce_loss(pred, label)
    
    def look_up(self, src, rel, ent_embeddings, rel_embeddings):        
        src_embed = torch.index_select(ent_embeddings, 0, src)
        rel_embed = torch.index_select(rel_embeddings, 0, rel)
        # obj_embed = ent_embeddings
        return src_embed, rel_embed
    
    def create_embeddings(self):
        self.embeddings = None

    def create_encoder(self):  ##
        self.encoder = None
    
    def create_decoder(self):
        self.decoder = None

    def gnn_forward(self, ent_embeddings, rel_embeddings):      
        
        if self.encoder is not None:
            return self.encoder(ent_embeddings, self.edge_index,
                                self.edge_type, rel_embeddings)
        else:
            return ent_embeddings, rel_embeddings
    
    def forward(self, batch_data):                  
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.to(self.device)
        src, rel = batch_data[:, 0], batch_data[:, 1]
        
        ent_embeddings, rel_embeddings = self.gnn_forward(*self.embeddings())

        src_embed, rel_embed= self.look_up(src, rel, ent_embeddings, rel_embeddings)
        
        features = self.decoder(src_embed, rel_embed, ent_embeddings)
        return features

    def save_embeddings(self):                          
        self.eval()
        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.gnn_forward(*self.embeddings())
            
            self.final_ent_embeddings.data = ent_embeddings
            self.final_rel_embeddings.data = rel_embeddings
        
        self.train()

    def predict(self, batch_data):              
        if not isinstance(batch_data, torch.Tensor):
            batch_data = torch.from_numpy(batch_data).to(self.device)
        src, rel = batch_data[:, 0], batch_data[:, 1]
        with torch.no_grad():
            ent_embeddings, rel_embeddings = self.final_ent_embeddings.data, self.final_rel_embeddings.data
            src_embed, rel_embed = self.look_up(src, rel, ent_embeddings, rel_embeddings)
            features = self.decoder(src_embed, rel_embed, ent_embeddings)
            if isinstance(features, tuple):
                return features[0]
            else:
                return features
        


class Embed_TransE(Model):
    
    def create_embeddings(self):            
        self.embeddings = Embeddings(self.num_ent, self.num_rel, self.p).to(self.device)
    
    def create_decoder(self):
        self.decoder = TransE(self.p, self.device).to(self.device)


class Embed_DistMult(Embed_TransE):
    
    def create_decoder(self):
        self.decoder = DistMult(self.p, self.device).to(self.device)


class Embed_ConvE(Embed_TransE):
    
    def create_decoder(self):           
        self.decoder = ConvE(self.num_ent, self.p, self.device).to(self.device)
