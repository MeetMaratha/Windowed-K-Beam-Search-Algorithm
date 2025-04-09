import os
import random
from pathlib import Path
from time import time
from typing import Any, Dict, List

import evaluate as eval
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim as optim
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from models.decoder import Decoder
from models.encoder import Encoder
from models.sequence_to_sequence import SequenceToSequence

# Set random seed for reproducibility
SEED: int = 0
BATCH_SIZE: int = 128
TRAIN_DATASET_SIZE: int = 100_000
VALIDATION_DATASET_SIZE: int = 3000
NUM_EPOCHS: int = 30
CLIP: float = 1.0
torch.manual_seed(SEED)
random.seed(SEED)


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


def train(
    model: SequenceToSequence,
    dataloader: DataLoader,
    optimzer: optim.AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    clip: int = 1,
) -> float:

    # Set model to training mode
    model.train()
    epoch_loss: float = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):

        # Move input and target sequences to the device
        german_data: torch.Tensor = batch["input"].to(device)
        english_data: torch.Tensor = batch["labels"].to(device)

        # Zero the gradients of the optimizer
        optimizer.zero_grad()

        # Pass the data through the model
        output: torch.Tensor = model(german_data, english_data, teacher_forcing=True)

        # Exclude first start of sequence token for computing loss
        output_dim: int = output.shape[-1]
        output = output[:, 1:, :].contiguous().view(-1, output_dim)
        target_sentence: torch.Tensor = english_data[:, 1:].contiguous().view(-1)

        # Compute the loss
        loss = criterion(output, target_sentence)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Add the epoch loss
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(
    model: SequenceToSequence,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    en_tokenizer,
) -> float:

    # Set model in the evaluation mode
    model.eval()
    epoch_loss: float = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):

            # Move the data to device
            german_tokenized: torch.Tensor = batch["input"].to(device)
            english_tokenized: torch.Tensor = batch["labels"].to(device)

            # Send the data through the model without teacher forcing (batch_size, target_sentence_length, target_sentence_vocab_length)
            output: torch.Tensor = model(
                german_tokenized, english_tokenized, teacher_forcing=False
            )

            log_probabilities = F.log_softmax(output, dim=-1)
            predicted_output = torch.zeros(
                size=(output.shape[0], output.shape[1]), dtype=torch.int64
            )

            # Remove first start of sequence token from computing loss
            for batch_id in range(output.shape[0]):
                for sequence_length_id in range(output.shape[1]):
                    predicted_output[batch_id, sequence_length_id] = log_probabilities[
                        batch_id, sequence_length_id, :
                    ].argmax(dim=-1)

            target_sentence: torch.Tensor = english_tokenized[:, 1:]

            # Compute the sacrebleu score
            # Convert the sequence to text
            predicted_english_sentence = en_tokenizer.batch_decode(
                predicted_output, skip_special_tokens=True
            )
            target_english_sentence = en_tokenizer.batch_decode(
                target_sentence, skip_special_tokens=True
            )
            score: Dict[str, Any] = criterion.compute(
                predictions=predicted_english_sentence,
                references=target_english_sentence,
            )
            epoch_loss += score["score"]

    return epoch_loss / len(dataloader)


def beamSearchDecoding(
    model: SequenceToSequence,
    input: torch.Tensor,
    de_tokenizer,
    en_tokenizer,
    beam_width: int = 5,
    max_length: int = 50,
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
    beams = [
        (
            [
                sos_token,
            ],
            0.0,
            hidden_layer,
            cell_layer,
        )
    ]

    for _ in range(max_length):
        computed_beams = []
        for sequence, score, hidden_layer, cell_layer in beams:
            if sequence[-1] in [eos_token, en_tokenizer.pad_token_id]:
                computed_beams.append(
                    (
                        sequence + [en_tokenizer.pad_token_id],
                        score,
                        hidden_layer,
                        cell_layer,
                    )
                )
                continue

            last_token = torch.tensor([sequence[-1]], device=device).unsqueeze(0)

            output_logits, new_hidden_layer, new_cell_layer = model.decoder(
                last_token, hidden_layer, cell_layer
            )

            log_probabilities = F.log_softmax(output_logits, dim=1).squeeze(0)

            top_log_probs, top_tokens = torch.topk(log_probabilities, beam_width)

            for log_prob, token in zip(top_log_probs, top_tokens):
                new_seq = sequence + [token.item()]
                new_score = score + log_prob.item()
                computed_beams.append(
                    (new_seq, new_score, new_hidden_layer, new_cell_layer)
                )

        beams = sorted(computed_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    return beams[0][0]


def evaluateBeamVersion(
    model: SequenceToSequence,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    de_tokenizer,
    en_tokenizer,
    beam_width: int = 5,
) -> float:

    # Set model to eval mode
    model.eval()
    epoch_loss: float = 0

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Evaluating using K Beams", leave=False):

            # Move the batch to the device
            input: torch.Tensor = batch["input"].to(device)
            target: torch.Tensor = batch["labels"].to(device)
            max_length: int = target.shape[1]

            # Generate predictions for each example in the batch using beam search

            batch_predictions: List[torch.Tensor] = []

            for i in range(input.size(0)):
                # (1, input_dim)
                input_sentence = input[i].unsqueeze(0)

                prediction: torch.tensor = beamSearchDecoding(
                    model=model,
                    input=input_sentence,
                    beam_width=beam_width,
                    max_length=max_length,
                    de_tokenizer=de_tokenizer,
                    en_tokenizer=en_tokenizer,
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
    dataset_path: Path = Path(".", "dataset", "tokenized_data")
    tokenized_dataset: DatasetDict = load_from_disk(dataset_path=dataset_path)

    # Get a subset of the dataset
    training_data: Dataset = (
        tokenized_dataset["train"].shuffle(SEED).select(range(TRAIN_DATASET_SIZE))
    )
    validation_data: Dataset = (
        tokenized_dataset["validation"]
        .shuffle(SEED)
        .select(range(VALIDATION_DATASET_SIZE))
    )
    test_data: Dataset = tokenized_dataset["test"].shuffle(SEED)

    # Generate dataloaders
    train_dataloader: DataLoader = DataLoader(
        dataset=training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_func,
    )
    valid_dataloader: DataLoader = DataLoader(
        dataset=validation_data, batch_size=BATCH_SIZE, collate_fn=collate_func
    )
    test_dataloader: DataLoader = DataLoader(
        dataset=test_data, batch_size=BATCH_SIZE, collate_fn=collate_func
    )

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

    # Make an optimizer
    optimizer = optim.AdamW(params=seq_to_seq.parameters())

    # Make a loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=english_tokenizer_pad_token)

    best_validation_loss: float = 0

    model_save_folder: Path = Path(".", "saved_models")

    if not model_save_folder.exists():
        os.mkdir(model_save_folder)

    for epoch in range(NUM_EPOCHS):
        start_time: float = time()

        train_loss: float = train(
            model=seq_to_seq,
            dataloader=train_dataloader,
            optimzer=optimizer,
            criterion=criterion,
            clip=CLIP,
            device=device,
        )
        valid_loss: float = evaluate(
            model=seq_to_seq,
            dataloader=valid_dataloader,
            criterion=eval.load("sacrebleu"),
            device=device,
            en_tokenizer=en_tokenizer,
        )

        end_time: float = time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        if valid_loss > best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(
                seq_to_seq.state_dict(),
                Path(
                    model_save_folder,
                    f"custom_model_normal_{epoch}_bleu_{valid_loss}.pt",
                ),
            )
            print(f"Model saved at epoch {epoch} with BLEU {valid_loss:.2f}")

        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}")

    # K-Beams implementation
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
    best_validation_loss: float = 0

    for epoch in range(NUM_EPOCHS):
        start_time: float = time()

        train_loss: float = train(
            model=seq_to_seq,
            dataloader=train_dataloader,
            optimzer=optimizer,
            criterion=criterion,
            clip=CLIP,
            device=device,
        )
        valid_loss: float = evaluateBeamVersion(
            model=seq_to_seq,
            dataloader=valid_dataloader,
            criterion=eval.load("sacrebleu"),
            device=device,
            en_tokenizer=en_tokenizer,
            de_tokenizer=de_tokenizer,
        )

        end_time: float = time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        if valid_loss > best_validation_loss:
            best_validation_loss = valid_loss
            torch.save(
                seq_to_seq.state_dict(),
                Path(
                    model_save_folder,
                    f"custom_model_k_beams_{epoch}_bleu_{valid_loss}.pt",
                ),
            )
            print(f"Model saved at epoch {epoch} with BLEU {valid_loss:.2f}")

        print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f}")
