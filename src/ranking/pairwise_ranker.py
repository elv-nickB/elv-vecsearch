import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Dict, Tuple, Iterable

from src.query_processing.simple import SimpleQueryProcessor
from src.ranking.rank import Ranker
from src.index.faiss import Index

class PairwiseRanker(Ranker):
    """
    A simple ranker that uses a scorer to rank documents based on their similarity to the query
    
    Use this class when scoring purely based off a query embedding and a vector document
    """
    def __init__(self, index: Index, model_path: str):
        self.index = index
        self.model = SimCLR_with_merge(768, 512, 256, [
                "Object Detection",
                "Celebrity Detection",
                "Logo Detection",
                "Landmark Recognition",
                "Optical Character Recognition",
                "Segment Labels",
                "Action Detection",
                "Speech to Text",
                "Display Title",
            ], mode="stack")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def rank(self, uids: Iterable[str], limit: int, query: SimpleQueryProcessor.ProcessedQuery) -> List[Tuple[str, float]]:
        scores = []
        for uid in uids:
            _, _, q, d = self.model.forward(torch.from_numpy(query["embedding"]).unsqueeze(0), self._get_embeddings(uid))
            score = torch.nn.functional.cosine_similarity(q, d).item()
            scores.append((score, uid))
        return [(uid, score) for score, uid in sorted(scores, reverse=True)][:limit]

    def _get_embeddings(self, uid: str) -> torch.Tensor:
        embeds = self.index.get_embeddings(uid)
        tracks = ['f_object', 'f_celebrity', 'f_logo', 'f_landmark', 'f_characters', 'f_segment', 'f_action', 'f_speech_to_text', 'f_display_title']
        input = []
        for track in tracks:
            if track in embeds:
                input.append(torch.sum(torch.from_numpy(embeds[track]), dim=0).squeeze(0))
            else:
                input.append(torch.zeros(768))
        return torch.stack(input).unsqueeze(0)
    
def stack_feature(feature:Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    # each feature is [D, ] shape
    return torch.stack([feature[k] for k in keys], dim=0)

class Merge(nn.Module):
    def __init__(self, keys:List[str]):
        super(Merge, self).__init__()
        # for k in keys:
        #     self.register_parameter(k, nn.Parameter(torch.rand(1), requires_grad=True))
        self.alphas = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(1 / len(keys), requires_grad=True)) for k in keys
        })
    
    def forward(self, features:Dict[str, torch.Tensor]) -> torch.Tensor:       
        return torch.stack([features[k] * self.alphas[k] for k in features]).mean(dim=0)
    
class Merge_stacked(nn.Module):
    def __init__(self, n:int, cross_attn=True, query_dim=768):
        # n is num of features
        super(Merge_stacked, self).__init__()
        self.cross_attn = cross_attn
        if cross_attn:
            # we can expand this to a MLP layer
            self.fc = nn.Linear(query_dim, n)
        else:
            self.alphas = torch.nn.parameter(torch.rand(n, requires_grad=True))
    
    def forward(self, feature:torch.Tensor, query:Optional[torch.Tensor]=None) -> torch.Tensor:
        # feature shape: [B, n, _d] we do not car e
        # query shape : [B. feat_dim]
        if self.cross_attn:
            assert query is not None
            alphas = torch.softmax(self.fc(query).unsqueeze(1), dim=-1) # [N, 1, n]
            return (alphas @ feature).squeeze(1)
        else:
            return self.alphas @ feature
        
class Merge_flattened(nn.Module):
    def __init__(self, in_dim:int, out_dim:int):
        super(Merge_flattened, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x:torch.Tensor, query:Optional[torch.Tensor]=None) -> torch.Tensor:
        # x shape [B, n * feat_dim]
        return self.fc(x)
    
class Encoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(input_dim, hidden_dim, bias=False)),
                ("bn1", nn.BatchNorm1d(hidden_dim)),
                ("act1", nn.ReLU(inplace=True)),
                # this is simulating the resnet's final fc layer
                ("fc2", nn.Linear(hidden_dim, hidden_dim),),
                ("bn2",  nn.BatchNorm1d(hidden_dim)),
            ])
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class SimCLR_with_merge(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, proj_dim:int, keys:List[str], mode:Optional[str]=None, cross_attn:bool=True):
        super(SimCLR_with_merge, self).__init__()
        # define merge model
        if mode == "flat":
            self.merge = Merge_flattened(in_dim=input_dim*len(keys), out_dim=input_dim)
        elif mode == "stack":
            self.merge = Merge_stacked(n=len(keys), cross_attn=cross_attn, query_dim=input_dim)
        elif mode is None:
            self.merge = Merge(keys)
        else:
            raise NotImplementedError("Please select the merge method from [flat, stack, None]")
        
        # create the encoder
        self.encoder = Encoder(input_dim, hidden_dim)

        # Replace the fc layer with an Identity function

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            OrderedDict([
                ("fc1", nn.Linear(hidden_dim, proj_dim, bias=False)),
                ("act1", nn.ReLU(inplace=True)),
                ("fc2", nn.Linear(proj_dim, proj_dim, bias=False)),
            ])
        )

    def forward(self, queries:torch.Tensor, features:torch.Tensor, other_feature:Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.merge(features, queries)

        h_i = self.encoder(queries)
        h_j = self.encoder(features)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        if other_feature is None:
            return h_i, h_j, z_i, z_j
    
        else:
            other_feature = self.merge(other_feature, queries)
            h_n = self.encoder(other_feature)
            z_n = self.projector(h_n)
            return h_i, h_j, h_n, z_i, z_j, z_n
