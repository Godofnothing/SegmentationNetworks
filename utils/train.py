import torch
from tqdm import tqdm

from utils import inference

def train_epoch(model, train_loader, optimizer, loss_fn, metric=None, device='cpu'):
    model.train()

    average_metrics = 0
    average_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        average_loss += loss.item()

        if metric:
            y_pred_probabilities = inference.get_probabilities(y_pred)
            y_pred_labels = inference.get_binary_outputs(y_pred_probabilities)
            with torch.no_grad():
                average_metrics += metric(y_pred_labels, y_batch).item()

    average_metrics = average_metrics / len(train_loader)
    average_loss = average_loss / len(train_loader)

    return average_loss, average_metrics


def val_epoch(model, val_loader, loss_fn, metric=None, device='cpu'):
    model.eval()

    average_metrics = 0
    average_loss = 0

    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        with torch.no_grad():
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

        average_loss += loss.item()

        if metric:
            average_metrics += metric(y_pred, y_batch).item()

    average_metrics = average_metrics / len(val_loader)
    average_loss = average_loss / len(val_loader)

    return average_loss, average_metrics


def run_training_loop(model, train_loader, val_loader, optimizer, loss_fn, num_epochs, metric=None, scheduler=None, device='cpu', verbose=False, save_best_model=False):

    history = {
        "train_loss" : [],
        "val_loss" : [],
        "train_metric" : [],
        "val_metric" : []
    }

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_metric = train_epoch(model, train_loader, optimizer, loss_fn, metric, device)
        val_loss, val_metric = val_epoch(model, val_loader, loss_fn, metric, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_metric"].append(train_metric)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if verbose:
            print(f"Epoch: {epoch}\nTrain loss: {train_loss:3f}\tVal loss: {val_loss:3f}")
            if metric:
                print(f"Train metric: {train_metric:3f}\tVal metric: {val_metric:3f}\n---")

    return history
