from typing import Tuple

import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:

        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        return None

    def forward(
        self,
        input_sentence_tokenized: torch.Tensor,
        hidden_layer: torch.Tensor,
        cell_layer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input_sentence_tokenized (torch.Tensor): The input sentence encoded till now. (batch_size, 1)
        hidden_layer (torch.Tensor): The hidden layer from that has been encoded till now. (num_layer, batch_size, hidden_dim)
        cell_layer (torch.Tensor): The hidden layer from that has been encoded till now. (num_layer, batch_size, hidden_dim)
        """

        # (batch_size, 1) -> (batch_size, 1, embedding_dim)
        embedded_layer: torch.Tensor = self.embedding(input_sentence_tokenized)

        # Output Dim: (batch_size, 1, hidden_dim)
        # Hidden Dim: (num_layers, batch_size, hidden_dim)
        # Cell Dim: (num_layers, batch_size, hidden_dim)
        output: torch.Tensor
        hidden: torch.Tensor
        cell: torch.Tensor
        output, (hidden, cell) = self.lstm(embedded_layer, (hidden_layer, cell_layer))

        # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        output = output.squeeze(1)

        # (batch_size, hidden_dim) -> (batch_size, output_size)
        prediction: torch.Tensor = self.fc_out(output)

        return (prediction, hidden, cell)


if __name__ == "__main__":

    batch_size: int = 4
    embedding_dim: int = 1024
    num_layers: int = 2
    hidden_dim: int = 1024
    vocab_size: int = 5
    input_sentnece_dim: int = 512
    decoder: Decoder = Decoder(
        output_dim=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    tokenized_input: torch.Tensor = torch.randint(
        low=0, high=vocab_size, size=(batch_size, 1)
    )
    prediction: torch.Tensor
    hidden: torch.Tensor = torch.randn(size=(num_layers, batch_size, hidden_dim))
    cell: torch.Tensor = torch.randn(size=(num_layers, batch_size, hidden_dim))
    prediction, hidden, cell = decoder(tokenized_input, hidden, cell)

    print("Input size: ", tokenized_input.shape)
    print("Prediction size: ", prediction.shape)
    print("Hidden layer shape: ", hidden.shape)
    print("Cell layer shape:", cell.shape)
    print("Output: ", prediction)
