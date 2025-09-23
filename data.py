from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

def get_input():
  text = ""

  with open('input.txt', 'r') as file:
    for line in file:
      text += line.strip()
      text += '\n'


  chars = sorted(list(set(text)))
  stoi = {k: v for v, k in enumerate(chars)}
  itos = {v: k for v, k in enumerate(chars)}
  return chars, text, stoi, itos

# stoi = {k: v for v, k in enumerate(chars)}
# itos = {v: k for v, k in enumerate(chars)}

def encode_data(input, stoi):
  return [stoi[char] for char in input]

def decode(input, itos):
  decoded_text = ''.join([itos[val] for val in input])
  return decoded_text

class TextDataset(Dataset):
  def __init__(self, encoded_dataset, block_size):
    self.encoded_dataset = encoded_dataset
    self.block_size= block_size

  def __len__(self):
    return len(self.encoded_dataset) - self.block_size - 1

  def __getitem__(self, idx):
    x = torch.tensor(self.encoded_dataset[idx: idx + self.block_size])
    y = torch.tensor(self.encoded_dataset[idx + 1: idx + self.block_size + 1])
    return x, y

def get_dataloaders(encoded_dataset, block_size=256, batch_size=256, train_size=0.95):
  train_dataset = TextDataset(encoded_dataset[: int(len(encoded_dataset) * train_size)], block_size)
  val_dataset = TextDataset(encoded_dataset[int(len(encoded_dataset) * train_size): ], block_size)

  train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
  val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

  return train_loader, val_loader