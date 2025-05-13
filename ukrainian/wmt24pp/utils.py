def process_docs(dataset):
    return dataset.filter(lambda example: example["is_bad_source"] == False)
