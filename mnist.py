import sys, os
import torch
from torchvision import datasets, transforms
import torchsm
from sacred import Experiment
from sacred.observers import MongoObserver

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ex = Experiment('icpram2019')

from utils import MovingAverage

class Net(torchsm.BaseLayer):
    def __init__(self, input, output, **kwargs):
        torchsm.BaseLayer.__init__(self, input, output)

        self.hidden_layers = kwargs["hidden_layers"] if "hidden_layers" in kwargs else 1
        self.hidden_dim = kwargs["hidden_dim"] if "hidden_dim" in kwargs else 10
        self.stig_dim = kwargs["stig_dim"]

        self.n_inputs = input
        self.n_outputs = output
        
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



        

@ex.config
def config():
    batch_size = 20
    lr = 0.001
    total_its = 10
    
    
    n_inputs = 28
    n_outputs = 10
    time_ticks = 28

    stig_hidden_layers = 1
    stig_hidden_dim = 20
    stig_dim = 15

    hidden_dim=10
    hidden_layers=1

    avg_window = 100

    use_mongo = False
    if use_mongo:
        ex.observers.append(MongoObserver.create())

@ex.capture
def preProcess(x, n_inputs, time_ticks):
    x = torch.tensor(x, dtype=torch.double, device=device)
    x = x.reshape(-1, time_ticks, n_inputs)
    th = 0.05
    x[x >= th] = 1
    x[x < th] = 0
    return x


@ex.capture
def init_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/tmp/mnist_data", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/tmp/mnist_data", train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()
            ])),batch_size=batch_size, shuffle=True)
    
    return (train_loader, test_loader)



@ex.capture
def getNet(n_inputs, stig_hidden_dim, stig_hidden_layers, n_outputs, stig_dim, hidden_layers, hidden_dim, batch_size):
    return Net(
        n_inputs, n_outputs,
        stig_hidden_dim=stig_hidden_dim, stig_hidden_layers=stig_hidden_layers,
        stig_dim=stig_dim, hidden_layers=hidden_layers,
        hidden_dim=hidden_dim, batch_size=batch_size
    ).to(device)



    
@ex.capture
def train(net, train_loader, _run, lr, total_its, time_ticks, avg_window):
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    avg_loss = MovingAverage(avg_window)
    avg_accuracy = MovingAverage(avg_window)
    
    for it in range(0, total_its):
        for epoch, (batch_data, batch_target) in enumerate(train_loader):
            data = preProcess(batch_data)
            out = None
            for i in range(0, data.shape[1]):
                out = net.forward(
                    torch.tensor(data[:,i],dtype=torch.float32,device=data.device)
                    )
            loss = loss_fn(out, batch_target.type(torch.long).to(out.device))            
            
            _, preds = out.max(1)
            acc = (preds == batch_target.to(device)).float().mean()
            overall_epoch = len(train_loader)*it + epoch
            print("epoch:",overall_epoch,"/",len(train_loader)*total_its,"batch loss:",loss.item(), "batch accuracy: ",acc.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            avg_accuracy += acc.item()

            _run.log_scalar("loss", float(avg_loss))
            _run.log_scalar("accuracy", float(avg_accuracy))

            net.reset()

@ex.capture
def test(net, test_loader, batch_size):
    rights = 0.0
    tots = 0.0
    for batch_data, batch_target in test_loader:
        data = preProcess(batch_data)
        out = None
        for i in range(0, data.shape[1]):
            out = net.forward(
                torch.tensor(data[:,i],dtype=torch.float32,device=data.device)
                )

        rights += (out.max(1)[1] == (batch_target.to(data.device))).sum().item()
        tots += batch_size
        net.reset()
    return rights/tots


@ex.automain
def main():
    train_loader, test_loader = init_loaders()
    net = getNet()
    train(net, train_loader)
    acc = test(net, test_loader)
    print("accuracy: ", acc)
    import pickle
    pickle.dump(net, open("results/stigmem_"+str(acc), "wb"))
    return acc
