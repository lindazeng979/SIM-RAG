import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import re

def load_data(train_path, val_path, test_path):
    try:
        raw_train = pd.read_csv(train_path)
        raw_val = pd.read_csv(val_path)
        raw_test = pd.read_csv(test_path)
        print('Using customized datasets.')
    except:
        print('No customized dataset provided.')
        
    return raw_train, raw_val, raw_test

def prepare_dataset(df, num_samples=None):
    instruction = "Instruction: Predict if the following answer to the question and context should be accepted, 1, or rejected, 0, based on the rationale."

    if num_samples is not None:
        df = df.head(num_samples)

    inputs = [
        f"{instruction}\n{task_content} \nAnswer: {answer}\nRationale: {rationale}"
        for answer, rationale, task_content in zip(df['Reasoner Answer'].tolist(), df['Reasoner Rationale'].tolist(), df['Reasoner Task Content'].tolist())
    ]

    data_dict = {
        "input": inputs,
        "label": [str(label) for label in df['Gate Verdict'].tolist()]
    }
    return Dataset.from_dict(data_dict)


def create_dataset_dict(train_set, val_set, test_set):
    return DatasetDict({
        'train': train_set,
        'val': val_set,
        'test': test_set
    })

def tokenize_data(dataset, tokenizer, max_source_length, max_target_length):
    def preprocess_function(sample, padding="max_length"):
        inputs = [item for item in sample["input"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        labels = tokenizer(sample["label"], max_length=max_target_length, padding=padding, truncation=True)
        
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True, remove_columns=['input', 'label'])

def get_data_collator(tokenizer, model):
    from transformers import DataCollatorForSeq2Seq
    return DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)