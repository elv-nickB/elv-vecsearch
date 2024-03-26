import torch
from torch import nn
from typing import List, Tuple, Iterable

from src.query_processing.simple import SimpleQueryProcessor
from src.ranking.rank import Ranker
from src.index.faiss import Index

class MLPRanker(Ranker):
    """
    A simple ranker that uses a scorer to rank documents based on their similarity to the query
    
    Use this class when scoring purely based off a query embedding and a vector document
    """
    def __init__(self, index: Index, model_path: str):
        self.index = index
        self.model = MLP()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def rank(self, uids: Iterable[str], limit: int, query: SimpleQueryProcessor.ProcessedQuery) -> List[Tuple[str, float]]:
        scores = []
        for uid in uids:
            scores.append((self.model(torch.from_numpy(query["embedding"]).unsqueeze(0), self._get_embeddings(uid)).item(), uid))
        return [(uid, score) for score, uid in sorted(scores, reverse=True)][:limit]

    def _get_embeddings(self, uid: str) -> torch.Tensor:
        embeds = self.index.get_embeddings(uid)
        tracks = ['f_object', 'f_celebrity', 'f_logo', 'f_landmark', 'f_characters', 'f_segment', 'f_action', 'f_speech_to_text']
        input = []
        for track in tracks:
            if track in embeds:
                input.append(torch.sum(torch.from_numpy(embeds[track]), dim=0).squeeze(0))
            else:
                input.append(torch.zeros(768))
        return torch.cat(input, dim=0).unsqueeze(0)

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