from __future__ import annotations
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.proj = nn.Conv1d(64, n_classes, kernel_size=1)

    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=2)
        logits = self.proj(h)
        return logits

class SimpleLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=0.0 if layers==1 else dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden, n_classes)

    def forward(self, x, lengths):
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits.transpose(1,2)

class CNNLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden, n_classes)

    def forward(self, x):
        h = self.conv(x)
        h = h.transpose(1,2)
        out, _ = self.lstm(h)
        logits = self.fc(out)
        return logits.transpose(1,2)
