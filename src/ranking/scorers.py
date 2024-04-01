
from typing import Dict
from sentence_transformers import util as ut

from src.vars import *
from src.index.abstract import VectorDocument
from src.query_processing.abstract import Query
from src.query_processing.word_weights import WordWeightsProcessor as wwp
from src.ranking.abstract import Scorer

# Get a scorer with weights for each field
#
# A fields score is taken as the max non-negative similarity between the query and any embedding in that field (0 if none exist)...
# The returned score is the weighted sum of the field scores
# TODO: if we introduce a cutoff > 0 for field scores we may get better results
def get_weighted_scorer(weights: Dict[str, float]) -> Scorer:
    def scorer(query: Query, doc: VectorDocument) -> float:
        field_scores = {}
        for field, embeds in doc.items():
            field_scores[field] = max(0, max(ut.dot_score(query["embedding"], embed).item() for embed in embeds))
        if len(field_scores) == 0:
            return 0
        return sum(weights[field]*score for field, score in field_scores.items())
    return scorer

# Get a scorer which uses /meta/search/weights from a index object to weight the fields while scoring a document
def get_word_weights_scorer() -> Scorer:
    def scorer(query: wwp.ProcessedQuery, doc: VectorDocument) -> float:
        ww_scorer = get_weighted_scorer(query["weights"])
        return ww_scorer(query, doc)
    return scorer

def get_semantic_scorer() -> Scorer:
    cat_thres = 0.0
    def scorer(query_embed: Query, doc: VectorDocument) -> float:
        query_embed = query_embed["embedding"]
        if len(doc) == 0: 
            return 0
        score_semantic = 0
        score_cat = 0
        for f, embeds in doc.items():
            if f in SEM_TRACKS:
                for e in embeds:
                    score_semantic = max((1.+ut.dot_score(query_embed, e).item())/2., score_semantic)
            elif f in CAT_TRACKS:
                for e in embeds:
                    score_cat = max((1.+ut.dot_score(query_embed, e).item())/2., score_cat)
        score_cat = 0 if score_cat < cat_thres else score_cat
        return (score_cat + score_semantic) / 2
    return scorer