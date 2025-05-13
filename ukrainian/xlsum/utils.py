try:
    from rouge_score import rouge_scorer, scoring
except ModuleNotFoundError as e:
    raise type(e)(
        "`rouge_score` is required for evaluating the model on XLSum (Ukrainian)."
    ) from e


ROUGE_SCORER = None


def process_docs(dataset):
    """
    Removes trailing whitespaces and adds newlines between sentences to correctly compute `rougeLsum`.
    """

    def _process_doc(doc):
        doc["text"] = " ".join(doc["text"].strip().split()).replace(" . ", ".\n")
        doc["summary"] = " ".join(doc["summary"].strip().split()).replace(" . ", ".\n")
        return doc

    return dataset.map(_process_doc)


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rougeLsum"]

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        # init RougeScorer once (https://github.com/EleutherAI/lm-evaluation-harness/issues/1692)--rouge_types are constant
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def process_results(doc, results):
    completion = results[0]
    reference = doc["summary"]
    rougeL = rouge([reference], [completion])
    return {"rougeL": rougeL}
