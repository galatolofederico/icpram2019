import sys, os
import torch
from torchvision import datasets, transforms
from sacred import Experiment
from sacred.observers import MongoObserver

from dataset.mnist_stroke import MNISTStroke
from utils import MovingAverage

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ex = Experiment('icpram2019_stroke')

from networks import *

@ex.config
def config():
    arch = "stigmergic"

    batch_size = 20
    lr = 0.001
    total_its = 10
    
    n_inputs = 4
    n_outputs = 10

    if arch == "stigmergic":
        stig_hidden_layers = 1
        stig_hidden_dim = 20
        stig_dim = 30

        hidden_dim=20
        hidden_layers=1

    if arch == "lstm":
        hidden_size = 20
        num_layers = 1
    
    if arch == "recurrent":
        recurrent_dim = 30
        recurrent_hidden = 50
        hidden_dim = 50

    avg_window = 100

    use_mongo = False
    if use_mongo:
        ex.observers.append(MongoObserver.create())


@ex.capture
def init_loaders(batch_size):
    train_loader = torch.utils.data.DataLoader(
        MNISTStroke(
            "/tmp/mnist_stroke", train=True),
        batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        MNISTStroke(
            "/tmp/mnist_stroke", train=False),
        batch_size=batch_size, shuffle=True
    )
    
    return (train_loader, test_loader)


@ex.capture
def getStigmergicNet(n_inputs, stig_hidden_dim, stig_hidden_layers, n_outputs, stig_dim, hidden_layers, hidden_dim, batch_size):
    return StigmergicNet(
        n_inputs, n_outputs,
        stig_hidden_dim=stig_hidden_dim, stig_hidden_layers=stig_hidden_layers,
        stig_dim=stig_dim, hidden_layers=hidden_layers,
        hidden_dim=hidden_dim, batch_size=batch_size
    ).to(device)


@ex.capture
def getLSTMNet(n_inputs, n_outputs, hidden_size, num_layers):
    return LSTMNet(
        n_inputs, n_outputs, hidden_size, num_layers=num_layers
    ).to(device)


@ex.capture
def getRecurrentNet(n_inputs, n_outputs, recurrent_dim, recurrent_hidden, hidden_dim):
    return RecurrentNet(
        n_inputs, n_outputs, recurrent_dim, recurrent_hidden=recurrent_hidden, classification_hidden=hidden_dim
    ).to(device)


@ex.capture
def getNet(arch):
    if arch == "stigmergic":
        return getStigmergicNet()
    if arch == "lstm":
        return getLSTMNet()
    if arch == "recurrent":
        return getRecurrentNet()
    else:
        raise Exception("Unknown architecture: %s" % arch)


    
@ex.capture
def train(net, train_loader, _run, lr, total_its, avg_window, batch_size, n_outputs):
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    avg_loss = MovingAverage(avg_window)
    avg_accuracy = MovingAverage(avg_window)
    
    for it in range(0, total_its):
        for epoch, (batch_data, batch_target) in enumerate(train_loader):
            data = batch_data.to(device)
            outs = torch.zeros(batch_size, n_outputs).to(device)

            for i in range(0, data.shape[1]):
                outs += net.forward(
                    torch.tensor(data[:,i],dtype=torch.float32,device=data.device)
                    ) * (data[:, i, 3] == 1)[:, None].type(torch.float).to(data.device)

            loss = loss_fn(outs, batch_target.type(torch.long).to(outs.device))            
            
            _, preds = outs.max(1)
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
def test(net, test_loader, batch_size, n_outputs):
    rights = 0.0
    tots = 0.0
    for batch_data, batch_target in test_loader:
        data = batch_data.to(device)
        outs = torch.zeros(batch_size, n_outputs).to(device)

        for i in range(0, data.shape[1]):
            outs += net.forward(
                torch.tensor(data[:,i],dtype=torch.float32,device=data.device)
                ) * (data[:, i, 3] == 1)[:, None].type(torch.float).to(data.device)

        rights += (outs.max(1)[1] == (batch_target.to(data.device))).sum().item()
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
    pickle.dump(net, open("results/stigmem_stroke_"+str(acc), "wb"))
    return acc
