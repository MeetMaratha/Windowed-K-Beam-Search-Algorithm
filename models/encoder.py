from typing import Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int
    ) -> None:

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        return None

    def forward(
        self, input_tokenized: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_tokenized (torch.Tensor): Input tokenized sentence (batch_size, input_sentence_length)
        """

        # (batch_size, input_sentence_length) -> (batch_size, input_sentnece_length, embedding_dim)
        embedded_input = self.embedding(input_tokenized)

        # Output: (batch_size, input_sentnece_length, hidden_dim)
        # Hidden: (num_layers, batch_size, hidden_dim)
        # Cell: (num_layers, batch_size, hidden_dim)
        output, (hidden, cell) = self.lstm(embedded_input)

        return (hidden, cell)


if __name__ == "__main__":

    batch_size: int = 4
    embedding_dim: int = 1024
    num_layers: int = 2
    hidden_dim: int = 1024
    vocab_size: int = 2048
    input_sentnece_dim: int = 512
    encoder: Encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    tokenized_input: torch.Tensor = torch.randint(
        low=0, high=vocab_size, size=(batch_size, input_sentnece_dim)
    )
    hidden: torch.Tensor
    cell: torch.Tensor
    hidden, cell = encoder(tokenized_input)

    print("Input size: ", tokenized_input.shape)
    print("Hidden layer shape: ", hidden.shape)
    print("Cell layer shape:", cell.shape)
