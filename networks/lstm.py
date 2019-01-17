import torch

class LSTMNet(torch.nn.Module):
    def __init__(self, inputs, outputs, hidden_size, **kwargs):
        torch.nn.Module.__init__(self)

        self.lstm = torch.nn.LSTM(inputs, hidden_size, **kwargs)
        self.fc = torch.nn.Linear(hidden_size, outputs)

        self.hidden = None
    
    def forward(self, x):
        out, self.hidden = self.lstm(x.view(1, x.shape[0], x.shape[1]), self.hidden)
        return self.fc(out[0])
    
    def reset(self):
        self.hidden = None
    
    def to(self, *args, **kwargs):
        self = torch.nn.Module.to(self, *args, **kwargs)

        self.lstm = self.lstm.to(*args, **kwargs)
        self.fc = self.fc.to(*args, **kwargs)

        return self