import torch

class RecurrentNet(torch.nn.Module):
    def __init__(self, inputs, outputs, recurrent_size, **kwargs):
        torch.nn.Module.__init__(self)
        self.recurrent_size = recurrent_size

        self.recurrent = torch.nn.Sequential(
            torch.nn.Linear(inputs+recurrent_size, kwargs["recurrent_hidden"]),
            torch.nn.PReLU(),
            torch.nn.Linear(kwargs["recurrent_hidden"], recurrent_size),
            torch.nn.PReLU()
        )

        self.classification = torch.nn.Sequential(
            torch.nn.Linear(recurrent_size, kwargs["classification_hidden"]),
            torch.nn.PReLU(),
            torch.nn.Linear(kwargs["classification_hidden"], outputs),
            torch.nn.PReLU()
        )

        self.reset()
    
    def forward(self, x):
        if self.hidden is None:
            self.hidden = torch.zeros(x.shape[0], self.recurrent_size, device=x.device)
            
        self.hidden = self.recurrent(torch.cat((x, self.hidden), dim=1))
        return self.classification(self.hidden)
    
    def reset(self):
        self.hidden = None
    
    def to(self, *args, **kwargs):
        self = torch.nn.Module.to(self, *args, **kwargs)

        if self.hidden is not None: self.hidden = self.hidden.to(*args, **kwargs)
        self.recurrent = self.recurrent.to(*args, **kwargs)
        self.classification = self.classification.to(*args, **kwargs)

        return self