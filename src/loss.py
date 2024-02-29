import numpy as np
from sklearn.metrics import dcg_score as sk_dcg_score


def dcg_at_k(true_rating, pred_rank, pred_score, topk=20, useRank=True):
    """_summary_

    Args:
        true_rating (numpy.ndarray): average rating 
        pred_rank (numpy.ndarray): zero indexed ranking 
        pred_score (numpy.ndarray): similarity score from search api
        topk (int, optional): _description_. Defaults to 20.
        useRank (bool, optional): use the rank created by search api. Defaults to True.

    Returns:
        float: The DCG score calculated up to the kth item.
    """
    if useRank:
        topk = min(topk, len(true_rating))
        indices = np.argsort(pred_rank)[:topk]
        discounts = np.log2(np.arange(1, topk + 1) + 1)
        dcg = np.sum((2**true_rating[indices] - 1) / discounts)
        return dcg
    else:
        return sk_dcg_score(true_rating, pred_score, k=topk, ignore_ties=True)


def pointwise_regression_error(true_rating, pred_score, pred_rank, topk=20, reRank=False, useRank=True):
    """_summary_
the pointwise approach [eq. 5.2] in learning to rank from Learning to Rank for Information Retrieval
By Tie-Yan Liu
    Args:
        true_rating (numpy.ndarray): _description_
        pred_score (numpy.ndarray): _description_
        pred_rank (numpy.ndarray): _description_
        topk (int, optional): _description_. Defaults to 20.
        useRank (bool, optional): use the rank created by search api. Defaults to True.
        reRank (bool, optional): normalize rank to become consecutive integers. Defaults to False.

    Returns:
        float: bounded regression loss
    """
    sorted_ratings = sorted(true_rating, reverse=True)
    idcg = dcg_at_k(sorted_ratings, pred_rank,
                    pred_score, topk, useRank=useRank)
    z_m = 1/idcg
    topk = min(topk, len(true_rating))
    if reRank:
        discount_sum = (
            2*sum((np.log2(np.arange(1, topk + 1) + 1)**(-2))))**0.5
    else:
        discount_sum = (2*sum((np.log2(pred_rank + 2)**(-2))))**0.5
    square_error_sum = (np.sum((pred_score - true_rating)**2))**0.5

    loss = z_m * discount_sum * square_error_sum
    return loss
