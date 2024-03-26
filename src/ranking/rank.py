
import numpy as np
from typing import List, Dict, Callable, Tuple, Iterable
import numpy as np
from sentence_transformers import util as ut

from src.index.abstract import Index, VectorDocument
from src.query_processing.simple import SimpleQueryProcessor

from src.ranking.abstract import Ranker

# function taking (query_embedding, VectorDocument) -> score
Scorer = Callable[[np.ndarray, VectorDocument], float] 

class SimpleRanker(Ranker):
    """
    A simple ranker that uses a scorer to rank documents based on their similarity to the query
    
    Use this class when scoring purely based off a query embedding and a vector document
    """
    def __init__(self, index: Index):
        self.index = index

    def rank(self, uids: Iterable[str], limit: int, query: SimpleQueryProcessor.ProcessedQuery) -> List[Tuple[str, float]]:
        scorer = self.get_weighted_scorer(query["weights"])
        scores = []
        for uid in uids:
            scores.append((scorer(query["embedding"], self.index.get_embeddings(uid)), uid))
        return [(uid, score) for score, uid in sorted(scores, reverse=True)][0:limit]

    # Get a scorer with weights for each field
    #
    # A fields score is taken as the max non-negative similarity between the query and any embedding in that field (0 if none exist)...
    # The returned score is the weighted sum of the field scores
    # TODO: if we introduce a cutoff > 0 for field scores we may get better results
    def get_weighted_scorer(self, weights: Dict[str, float]) -> Scorer:
        def scorer(query_embed: np.ndarray, doc: VectorDocument) -> float:
            field_scores = {}
            for field, embeds in doc.items():
                field_scores[field] = max(0, max(ut.dot_score(query_embed, embed).item() for embed in embeds))
            if len(field_scores) == 0:
                return 0
            return sum(weights[field]*score for field, score in field_scores.items())
        return scorer