import random
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import evaluate as eval
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from models.decoder import Decoder
from models.encoder import Encoder
from models.sequence_to_sequence import SequenceToSequence

# Set random seed for reproducibility
SEED: int = 0
TEST_BATCH_LIMIT: int = 600
BATCH_SIZE: int = 128
TRAIN_DATASET_SIZE: int = 100_000
VALIDATION_DATASET_SIZE: int = 3000
NUM_EPOCHS: int = 30
CLIP: float = 1.0
torch.manual_seed(SEED)
random.seed(SEED)


@dataclass
class Beam:
    """Class to represent a beam in the k-beams search."""

    sequence: field(default_factory=List[int])  # Token IDs in the sequence
    score: float  # Cumulative log probability score
    window_size: int  # Current window size before dropping
    hidden_layer: torch.Tensor = field(default_factory=torch.Tensor)
    cell_layer: torch.Tensor = field(default_factory=torch.Tensor)

    def __str__(self):
        return f"Sequence: {self.sequence}, Score: {self.score}, Window: {self.window_size}"


def collate_func(batch) -> Dict[str, Any]:

    # Stack all the data
    input: torch.Tensor = torch.tensor(
        [data["input_ids"] for data in batch], dtype=torch.int64
    )
    attention_mask: torch.Tensor = torch.tensor(
        [data["attention_mask"] for data in batch]
    )
    labels: torch.Tensor = torch.tensor(
        [data["labels"] for data in batch], dtype=torch.int64
    )

    return {"input": input, "attention_mask": attention_mask, "labels": labels}


def customBeamSearchDecoding(
    model: SequenceToSequence,
    input: torch.Tensor,
    de_tokenizer,
    en_tokenizer,
    beam_width: int = 5,
    max_length: int = 50,
    window_size: int = 3,
) -> torch.Tensor:

    # Run the input sentence through the encoder
    hidden_layer: torch.Tensor
    cell_layer: torch.Tensor
    hidden_layer, cell_layer = model.encoder(input)

    # Get the start of the sequence and end of the sequence token
    sos_token: int = (
        en_tokenizer.bos_token_id
        if en_tokenizer.bos_token_id is not None
        else en_tokenizer.cls_token_id
    )
    eos_token: int = (
        en_tokenizer.eos_token_id
        if en_tokenizer.eos_token_id is not None
        else en_tokenizer.sep_token_id
    )

    # (1, target_length, output_dim)
    beams: List[Beam] = [
        Beam(
            sequence=[
                sos_token,
            ],
            score=0.0,
            window_size=window_size,
            hidden_layer=hidden_layer,
            cell_layer=cell_layer,
        )
    ]

    for _ in range(max_length):
        computed_beams: List[Beam] = []
        for beam in beams:
            if beam.sequence[-1] in [eos_token, en_tokenizer.pad_token_id]:
                computed_beams.append(
                    Beam(
                        sequence=beam.sequence + [en_tokenizer.pad_token_id],
                        score=beam.score,
                        hidden_layer=beam.hidden_layer,
                        cell_layer=beam.cell_layer,
                        window_size=beam.window_size,
                    )
                )
                continue

            last_token = torch.tensor([beam.sequence[-1]], device=device).unsqueeze(0)

            output_logits, new_hidden_layer, new_cell_layer = model.decoder(
                last_token, beam.hidden_layer, beam.cell_layer
            )

            log_probabilities = F.log_softmax(output_logits, dim=1).squeeze(0)

            top_log_probs, top_tokens = torch.topk(log_probabilities, beam_width)

            for log_prob, token in zip(top_log_probs, top_tokens):
                new_seq = beam.sequence + [token.item()]
                new_score = beam.score + log_prob.item()
                computed_beams.append(
                    Beam(
                        sequence=new_seq,
                        score=new_score,
                        hidden_layer=new_hidden_layer,
                        cell_layer=new_cell_layer,
                        window_size=beam.window_size,
                    )
                )

        beams = sorted(computed_beams, key=lambda x: x.score, reverse=True)
        del computed_beams
        for beam in beams[:beam_width]:
            beam.window_size = window_size
        for beam in beams[beam_width:]:
            beam.window_size -= 1

        beams = [beam for beam in beams if beam.window_size > 0]

    return beams[0].sequence


def evaluateBeamVersion(
    model: SequenceToSequence,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    de_tokenizer,
    en_tokenizer,
    beam_width: int = 5,
    window_size: int = 3,
) -> float:

    # Set model to eval mode
    model.eval()
    epoch_loss: float = 0

    with torch.no_grad():

        for batch in tqdm(
            dataloader,
            desc=f"Evaluating using K Beams with window size: {window_size}",
            leave=False,
        ):

            # Move the batch to the device
            input: torch.Tensor = batch["input"].to(device)
            target: torch.Tensor = batch["labels"].to(device)
            max_length: int = target.shape[1]

            # Generate predictions for each example in the batch using beam search

            batch_predictions: List[torch.Tensor] = []

            for i in range(input.size(0)):
                # (1, input_dim)
                input_sentence = input[i].unsqueeze(0)

                prediction: torch.tensor = customBeamSearchDecoding(
                    model=model,
                    input=input_sentence,
                    beam_width=beam_width,
                    max_length=max_length,
                    de_tokenizer=de_tokenizer,
                    en_tokenizer=en_tokenizer,
                    window_size=window_size,
                )

                batch_predictions.append(prediction)
            batch_predictions = torch.tensor(batch_predictions)
            target_sentence: torch.Tensor = target[:, 1:]

            # Compute the sacrebleu score
            # Convert the sequence to text
            predicted_english_sentence = en_tokenizer.batch_decode(
                batch_predictions, skip_special_tokens=True
            )
            target_english_sentence = en_tokenizer.batch_decode(
                target_sentence[:, 1:], skip_special_tokens=True
            )
            score: Dict[str, Any] = criterion.compute(
                predictions=predicted_english_sentence,
                references=target_english_sentence,
            )
            epoch_loss += score["score"]

    return epoch_loss / len(dataloader)


if __name__ == "__main__":
    # Import the data
    print("Info: Importing Dataset...")
    dataset_path: Path = Path(".", "dataset", "tokenized_data")
    tokenized_dataset: DatasetDict = load_from_disk(dataset_path=dataset_path)
    print("Info: Dataset Imported!")

    print("Info: Generating Dataloader...")
    # Get the test data
    test_data: Dataset = (
        tokenized_dataset["test"].shuffle(SEED).select(range(TEST_BATCH_LIMIT))
    )
    # Generate dataloader
    test_dataloader: DataLoader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        collate_fn=collate_func,
        num_workers=12,
    )
    print("Info: Dataloader Generated!")

    print("Info: Loading Model...")
    # Get german words tokenizer vocab size
    de_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    german_vocab_size: int = de_tokenizer.vocab_size
    english_vocab_size: int = en_tokenizer.vocab_size
    english_tokenizer_pad_token: int = en_tokenizer.pad_token_id
    encoding_embedding_dim: int = 256
    decoding_embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the models
    encoder: Encoder = Encoder(
        vocab_size=german_vocab_size,
        embedding_dim=encoding_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    decoder: Decoder = Decoder(
        output_dim=english_vocab_size,
        embedding_dim=encoding_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    seq_to_seq: SequenceToSequence = SequenceToSequence(
        encoder=encoder, decoder=decoder, device=device
    )

    # Load the model
    seq_to_seq.load_state_dict(
        torch.load(
            "./saved_models/custom_model_normal_26_bleu_0.88331271267714.pt",
            weights_only=True,
        )
    )
    print("Info: Model Loaded!")

    print("Info: Testing Model on Standard K-Beams Search...")
    start_time: float = time()

    window_sizes: List[int] = [2, 3, 4, 5]
    scores: Dict[int, float] = dict()

    for window_size in window_sizes:
        test_score: float = evaluateBeamVersion(
            model=seq_to_seq,
            dataloader=test_dataloader,
            criterion=eval.load("sacrebleu"),
            device=device,
            en_tokenizer=en_tokenizer,
            de_tokenizer=de_tokenizer,
            window_size=window_size,
        )

        end_time: float = time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        scores[window_size] = test_score
        print(
            f"Testing Time for window size {window_size}: {int(epoch_mins)}m {int(epoch_secs)}s"
        )
        print(f"Scored: {test_score}")

    print("Info: Testing Finished!")
    print("\t Test SacreBLEU Scores:")
    for window_size, score in scores.items():
        print(f"\t\tWindow Size: {window_size}")
        print(f"\t\tScore: {score}")
