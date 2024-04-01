import torch
from torch import nn

from src.query_processing.simple import SimpleProcessor
from src.ranking.abstract import Scorer
from src.index.abstract import VectorDocument        

def get_mlp_scorer(model_path: str) -> Scorer:
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    def scorer(query: SimpleProcessor.ProcessedQuery, doc: VectorDocument) -> float:  
        input = []
        for track in model.tracks:
            if track in doc:
                input.append(torch.sum(torch.from_numpy(doc[track]), dim=0).squeeze(0))
            else:
                input.append(torch.zeros(768))
        embeds = torch.cat(input, dim=0).unsqueeze(0)
        return model(torch.from_numpy(query["embedding"]).unsqueeze(0), embeds).item()
    
    return scorer

class MLP(torch.nn.Module):   # May be use Transformer Blocks!!
    """
    Multi-layer perceptron with single hidden layer.
    """
    def __init__(self,
                input_dim=6144,
                query_dim = 768,
                num_output=1, 
                num_hidden_layers=1, scale = 1):
        
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.num_output = num_output
        self.num_hidden_layers = num_hidden_layers
        # tracks embeddings are ordered like so 
        self.tracks = ['f_object', 'f_celebrity', 'f_logo', 'f_landmark', 'f_characters', 'f_segment', 'f_action', 'f_speech_to_text']

        self.ip_layer = nn.Sequential(nn.Linear(self.input_dim, self.query_dim), nn.ReLU(True))
        
        inout_dims = [2*query_dim]
        for i in range(self.num_hidden_layers): inout_dims.append(inout_dims[i]//scale)
        hidden_blocks = [linearLeakyReluBlock(inout_dims[i], inout_dims[i+1]) for i in range(len(inout_dims)-1)]

        self.hidden_layers = nn.Sequential(*hidden_blocks)

        self.output =  nn.Linear(inout_dims[-1], self.num_output)

    def forward(self, query, input, feat = False):
        in_features = self.ip_layer(input)

        if feat: return in_features
        
        joint_feature = torch.hstack((in_features,query))

        hidden_features = self.hidden_layers(joint_feature)
        output = self.output(hidden_features)
        return output
        
def linearLeakyReluBlock(in_f, out_f):
    return nn.Sequential(nn.Linear(in_f, out_f),
                         nn.BatchNorm1d(out_f),
                        nn.LeakyReLU(True))