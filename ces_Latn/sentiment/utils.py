from sklearn.metrics import f1_score, confusion_matrix


def aggregate_macro_f1_score(items, **kwargs):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average='macro')
    return fscore
