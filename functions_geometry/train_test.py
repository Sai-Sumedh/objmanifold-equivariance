import torch
import math
from inspect import signature

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def train(dataloader, model, loss_fn, optimizer, status_update_freq=20, track_accuracy=True):
    """
    Runs one epoch of training using mini-batches
    Returns the evolution of training loss over mini-batches
    """
    size = len(dataloader.dataset)
    model.train()
    train_loss_trend = []
    train_acc_trend = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred=model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_trend.append(loss.item())
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        acc = 100*correct/len(y)
        train_acc_trend.append(acc)

        if batch % status_update_freq == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}, accuracy: {acc:>2f}  [{current:>5d}/{size:>5d}]")
            
    return train_loss_trend, train_acc_trend


def test(dataloader, model, loss_fn):
    """
    Evaluates model on test inputs using mini-batches
    Returns average test loss over mini-batches
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, 100*correct