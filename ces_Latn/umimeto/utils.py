


ANSWER_LETTERS = [chr(ord('A') + i) for i in range(ord('Z') - ord('A') + 1)]


def mmlu_get_choice(dataset):
    choice = [c for c in ANSWER_LETTERS if c in dataset.keys()]
    if len(choice) == 0:
        raise ValueError(f"No answer columns found in dataset")
    return choice


def mmlu_get_answer_index(dataset):
    return ANSWER_LETTERS.index(dataset["correct_answer"])


def mmlu_get_question_text_umimeto(dataset):
    """
    Umimeto has really bad questions, which require the topic at minimum to make sense
    """
    dataset_answer_keys = mmlu_get_choice(dataset)
    question_text = dataset['question'].strip()
    choices_text = "\n".join(f"{c}. {dataset[c]}" for c in dataset_answer_keys)
    topic = dataset["topic"]
    return f"{topic}: {question_text}\n{choices_text}\nOdpověď: "
