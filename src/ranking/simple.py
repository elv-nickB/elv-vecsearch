
from typing import Iterable, List, Tuple

from src.ranking.abstract import Ranker
from src.ranking.scorers import Scorer
from src.index.faiss import Index
from src.query_processing.abstract import Query

class SimpleRanker(Ranker):
    """
    A simple ranker that uses a given scorer to rank documents based on their similarity to the query
    
    Use this class when scoring purely based off a query embedding and a vector document
    """
    def __init__(self, index: Index, scorer: Scorer):
        self.index = index
        self.scorer = scorer

    def rank(self, uids: Iterable[str], limit: int, query: Query) -> List[Tuple[str, float]]:
        scores = []
        for uid in uids:
            scores.append((self.scorer(query, self.index.get_embeddings(uid)), uid))
        return [(uid, score) for score, uid in sorted(scores, reverse=True)][0:limit]