
from typing import Dict
from elv_client_py import ElvClient
from sentence_transformers import util as ut
import torch

from src.classes import VectorDocument, Scorer, ScorerFactory

def get_term_weight_scoring_factory(index_qid: str, client: ElvClient) -> ScorerFactory:
    def scorer_factory(query: str) -> Scorer:
        weights = get_term_weights_from_query(query, index_qid, client)
        return get_weighted_scorer(weights)
    return scorer_factory

def get_weighted_scorer(weights: Dict[str, float]) -> Scorer:
    def scorer(query_embed: torch.Tensor, doc: VectorDocument) -> float:
        field_scores = {}
        for field, embeds in doc.items():
            if embeds is not None:
                field_scores[field] = max(ut.dot_score(query_embed, embed).item() for embed in embeds)
        if len(field_scores) == 0:
            return 0
        return sum(weights[field]*score for field, score in field_scores.items())
    return scorer

# Args:
#   query: text query
#   word_weights: WordWeights type, obtained from download_word_weights or download_word_ratio_weights
#
# Returns:
#   Dict[str, float] mapping from field to weight for the given query
#
# Used for dynamically assigning weights to a query 
def get_term_weights_from_query(query: str, content_id: str, client: ElvClient) -> Dict[str, float]:
    terms = query.split(' ')
    term_weights = client.content_object_metadata(object_id=content_id, metadata_subtree='search/weights', select=terms)
    fields = list(next(iter(term_weights.values())).keys())
    wt = {f: 0 for f in fields}
    for term in term_weights:
        if term not in term_weights:
            continue
        for field in wt:
            wt[field] += term_weights[term][field]
    return wt