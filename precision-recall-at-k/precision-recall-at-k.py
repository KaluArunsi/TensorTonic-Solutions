def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    set_relevant, set_recommended = set(relevant), set(recommended[:k])
    hits = set_relevant.intersection(set_recommended)

    precision = len(hits) / k
    recall = len(hits) / len(relevant)

    return [precision, recall]