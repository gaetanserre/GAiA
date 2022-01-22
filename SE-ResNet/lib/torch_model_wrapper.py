# Created by Serré Gaëtan
# gaetan.serre93@gmail.com

import torch
import numpy as np

class TorchWrapper():
  def __init__(self, nn, device, optimizer, loss_function, metric=None):
    self.nn = nn
    self.device = device
    self.optimizer = optimizer
    self.loss_function = loss_function
    self.metric = metric

  @staticmethod
  def print_cuda_memory_state():
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    r = torch.cuda.memory_reserved(0) / 1024**3
    a = torch.cuda.memory_allocated(0) / 1024**3
    f = t - (r+a)
    print(f"Total mem: {t:.2f} GiB, Reserved mem: {r:.2f} GiB, Allocated mem: {a:.2f} GiB, Free mem: {f:.2f} GiB")
  
  @staticmethod
  def data_to_loader(data, batch_size, num_workers, shuffle):
    loader = torch.utils.data.DataLoader(
              data,
              batch_size=batch_size,
              shuffle=shuffle,
              num_workers=num_workers)
    return loader
  
  def predict(self, X, batch_size=32, num_workers=1):
    self.nn.eval()
    data = torch.from_numpy(X).float()
    loader = self.data_to_loader(data, batch_size, num_workers, shuffle=False)

    preds = None
    for data in loader:
      pred = self.nn(data.to(self.device))
      pred = pred.cpu().detach().numpy()
      del data
      if preds is not None:
        preds = np.concatenate((preds, pred))
      else:
        preds = pred
    return preds
  
  def fit(self, X, Y,
          valid_data=None,
          epochs=1,
          batch_size=32,
          num_workers=1,
          verbose=True,
          shuffle=True):
          
    self.nn.train()
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    loader = self.data_to_loader(train_set, batch_size, num_workers, shuffle=shuffle)

    history = {"loss": [], "val_loss": [], "metric": [], "val_metric": []}

    for epoch in range(epochs):  # loop over the dataset multiple times
      running_loss = 0.0
      metric = 0
      count = 0
      for data in loader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(self.device), data[1].to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.nn(inputs)
        loss = self.loss_function(outputs, labels)
        loss.backward()
        self.optimizer.step()

        if self.metric:
          metric += self.metric(labels, outputs)

        # print statistics
        running_loss += loss.item()
        count += 1

      loss = running_loss / count
      history["loss"].append(loss)

      if self.metric is not None:
        train_metric = metric / count
        history["metric"].append(train_metric.cpu().detach().numpy())

      if verbose:
        print(f"Epoch: {epoch+1}/{epochs} Loss: {loss:.2f}", end="")

        if self.metric is not None:
          print(f" Metric: {train_metric:.2f}", end="")
        
        if valid_data:
          X_valid, Y_valid = valid_data
          Y_valid = torch.from_numpy(Y_valid)
          preds = torch.from_numpy(self.predict(X_valid, batch_size=batch_size))

          if self.metric is not None:
            valid_metric = self.metric(Y_valid, preds)
            history["val_metric"].append(valid_metric.cpu().detach().numpy())

          valid_loss = self.loss_function(preds, Y_valid)
          history["val_loss"].append(valid_loss)

          print(f" Validation loss: {valid_loss:.2f}", end="")
          if self.metric is not None:
            print(f" Validation metric: {valid_metric:.2f}")
          else: print("")

          self.nn.train()
        else: print("")
      
    return history

  def save(self, filename):
    torch.save(self.nn, filename)
  
  def load(self, filename):
    self.nn = torch.load(filename)
  
  def get_parameters(self, trainable=False):
    return sum(p.numel() for p in self.nn.parameters() if not trainable or p.requires_grad)
