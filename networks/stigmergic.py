import torchsm
import torch

class StigmergicNet(torchsm.BaseLayer):
    def __init__(self, input, output, **kwargs):
        torchsm.BaseLayer.__init__(self, input, output)

        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 1
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_dim" in kwargs else 10
        self.stig_dim = kwargs["stig_dim"]

        self.n_inputs = input
        
        self.stigmergic_memory = torchsm.RecurrentStigmergicMemoryLayer(
            self.n_inputs, self.stig_dim,
            hidden_dim=kwargs["stig_hidden_dim"], hidden_layers=kwargs["stig_hidden_layers"],
        )


        self.classification_layer = torch.nn.Sequential()

        if self.hidden_layers != 0:
            self.classification_layer.add_module("input_w", torch.nn.Linear(self.stig_dim, self.hidden_dim))
            self.classification_layer.add_module("input_s", torch.nn.PReLU())
            
            for i in range(0, self.hidden_layers-1):
                self.classification_layer.add_module("l"+str(i)+"_w", torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.classification_layer.add_module("l"+str(i)+"_s", torch.nn.PReLU())
            
            self.classification_layer.add_module("output_w", torch.nn.Linear(self.hidden_dim, output))
            self.classification_layer.add_module("output_s", torch.nn.PReLU())
        else:
            self.classification_layer.add_module("linear", torch.nn.Linear(self.n_inputs, output))
            self.classification_layer.add_module("output_s", torch.nn.PReLU())
    
    def forward(self, input):
        return self.classification_layer(
            self.stigmergic_memory(input)
        )

    def reset(self):
        self.stigmergic_memory.reset()
    

    def to(self, *args, **kwargs):
        self = torchsm.BaseLayer.to(self, *args, **kwargs)
        
        self.stigmergic_memory = self.stigmergic_memory.to(*args, **kwargs)
        self.classification_layer = self.classification_layer.to(*args, **kwargs)
        return self
