import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoTokenizer


def preprocessData(
    sample: Dict[str, Any], tokenizer_de, tokenizer_en, max_length: int
) -> Dict[str, Any]:

    # Get German and English text from the sample
    source_text: List[List[str]] = [entry["de"] for entry in sample["translation"]]
    target_text: List[str] = [entry["en"] for entry in sample["translation"]]

    # Tokenize the sentences
    tokenized_input: List[List[int]] = tokenizer_de(
        source_text, padding="max_length", truncation=True, max_length=max_length
    )
    tokenized_target: List[List[int]] = tokenizer_en(
        target_text, padding="max_length", truncation=True, max_length=max_length
    )

    return {
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
        "labels": tokenized_target["input_ids"],
    }


if __name__ == "__main__":

    # Get raw dataset
    raw_dataset = load_dataset("wmt14", "de-en")

    print("Info: Dataset Loaded!")

    # Get the tokenizer for the data
    tokenizer_de = AutoTokenizer.from_pretrained("bert-base-german-cased")
    tokenizer_en = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Info: Starting tokenization of dataset.")

    tokenized_dataset = raw_dataset.map(
        preprocessData,
        batched=True,
        fn_kwargs={
            "tokenizer_de": tokenizer_de,
            "tokenizer_en": tokenizer_en,
            "max_length": 50,
        },
    )

    print("Info: Raw data tokenized!")
    print("Info: Storing dataset for future use.")

    dataset_path: Path = Path(".", "dataset")
    if not dataset_path.exists():
        os.mkdir(dataset_path)
    dataset_path = Path(dataset_path, "tokenized_data")

    tokenized_dataset.save_to_disk(dataset_path)

    print("Info: Stored dataset. Exiting function.")
