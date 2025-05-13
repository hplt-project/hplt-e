def reformat(doc):
    doc["choices"] = {
        "label": [item["marker"] for item in doc["answers"]],
        "text": [item["text"] for item in doc["answers"]],
    }
    del doc["answers"]
    return doc


def filter_language_and_literature(dataset):
    return dataset.map(reformat).filter(
        lambda example: example["subject"] == "ukrainian-language-and-literature"
    )


def filter_history_of_ukraine(dataset):
    return dataset.map(reformat).filter(
        lambda example: example["subject"] == "history-of-ukraine"
    )
