


def cermat_get_choice(dataset):
    if len(dataset['choices']) == 4:
        return ["A", "B", "C", "D"]
    elif len(dataset['choices']) == 5:
        return ["A", "B", "C", "D", "E"]
    else:
        raise ValueError(f"Invalid number of choices: {len(dataset['choices'])}")
