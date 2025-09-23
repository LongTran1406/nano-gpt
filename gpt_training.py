import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from model import NanoGPT
from data import get_dataloaders, get_input, encode_data

chars, text, stoi, itos = get_input()

encoded_dataset = encode_data(input = text, stoi = stoi)
  
train_loader, val_loader = get_dataloaders(encoded_dataset)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_size = 0.95
val_size = 0.05
batch_size = 256
block_size = 256
embedding_dim = 384
head_size = 64
head_num = 6
num_blocks = 3
vocab_size = len(chars)
drop_out = 0.2

model = NanoGPT(embedding_dim, block_size, head_size, head_num, drop_out, num_blocks, vocab_size, device).to(device)

optimizer = optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),
    eps=1e-9
)

criterion = CrossEntropyLoss()

epochs = 1

context = torch.zeros((1, 1), dtype=torch.long, device=device)

train_loss = []
val_loss = []

for epoch in range(epochs):
    model.train()  # set model to training mode
    total_train_loss = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # forward pass
        output = model(x_batch)
        batch_size, block_size, vocab_size = output.shape

        # compute loss
        loss = criterion(
            output.view(batch_size * block_size, vocab_size),
            y_batch.view(-1)
        )
        total_train_loss += loss.item()

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

        # print training & validation loss every 100 batches
        if batch_idx % 100 == 0:
            train_loss.append(loss.item())
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {loss.item():.4f}')

            # validation loop
            model.eval()  # switch to evaluation mode
            total_val_loss = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    output_val = model(x_val)
                    batch_size_val, block_size_val, vocab_size_val = output_val.shape
                    loss_val_batch = criterion(
                        output_val.view(batch_size_val * block_size_val, vocab_size_val),
                        y_val.view(-1)
                    )
                    total_val_loss += loss_val_batch.item()
            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Val Loss: {avg_val_loss:.4f}')
            model.train()  # switch back to training mode

    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Epoch {epoch} completed. Average Train Loss: {avg_train_loss:.4f}')
    break

torch.save(model.state_dict(), "nanogpt_model_state.pth")
