import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, sequence_length, input_size]

        output, hidden = self.rnn(x)

        # Use the final time step
        last_output = output[:, -1, :]

        logits = self.fc(last_output)

        return logits

model = SimpleRNN(
    input_size=20,
    hidden_size=64,
    num_classes=5
)

sample_sequence_batch = torch.randn(32, 10, 20)
output = model(sample_sequence_batch)

print(output.shape)  # torch.Size([32, 5])
