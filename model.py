import torch
import torch.nn as nn
import torch.nn.functional as F
from data import decode, get_input

chars, text, stoi, itos = get_input()

class Head(nn.Module):
  def __init__(self, embedding_dim, head_size, drop_out, block_size, device):
    super().__init__()
    self.k = nn.Linear(in_features = embedding_dim, out_features = head_size)
    self.q = nn.Linear(in_features = embedding_dim, out_features = head_size)
    self.v = nn.Linear(in_features = embedding_dim, out_features = head_size)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device = device)))
    self.dropout = nn.Dropout(drop_out)

  def forward(self, x):
    # x: (batch_size, block_size, embedding_dim)
    T = x.shape[1]

    key = self.k(x) # (batch_size, block_size, head_size)
    query = self.q(x) # (batch_size, block_size, head_size)
    value = self.v(x) # (batch_size, block_size, head_size)

    wei = (query @ key.transpose(-2, -1)) / (key.shape[-1]**0.5) # (batch_size, block_size, block_size)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (batch_size, block_size, block_size)
    wei = F.softmax(input = wei, dim = -1) # (batch_size, block_size, block_size)
    wei = self.dropout(wei)

    return wei @ value # (batch_size, block_size, head_size)

class MultiHead(nn.Module):
  def __init__(self, embedding_dim, head_size, head_num, drop_out, block_size, device):
    super().__init__()
    self.heads = nn.ModuleList([Head(embedding_dim, head_size, drop_out, block_size, device) for _ in range(head_num)])
    self.linear = nn.Linear(head_size * head_num, embedding_dim)
    self.dropout = nn.Dropout(drop_out)
  def forward(self, x):
    # x: (batch_size, block_size, embedding_dim)
    x = torch.cat([head(x) for head in self.heads], dim = -1) # (batch_size, block_size, head_size * head_num)
    x = self.dropout(self.linear(x)) # (batch_size, block_size, embedding_dim)
    return x


class FeedForward(nn.Module):
  def __init__(self, embedding_dim, drop_out):
    super().__init__()
    self.linear = nn.Linear(embedding_dim, embedding_dim)
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(drop_out)

  def forward(self, x):
    # x (batch_size, block_size, embedding_dim)
    x = self.dropout(self.activation(self.linear(x))) # (batch_size, block_size, embedding_dim)
    return x

class Block(nn.Module):
  def __init__(self, embedding_dim, head_size, head_num, drop_out, block_size, device):
    super().__init__()
    self.head = MultiHead(embedding_dim, head_size, head_num, drop_out, block_size, device)
    self.layer_norm1 = nn.LayerNorm(normalized_shape = embedding_dim)
    self.ffwd = FeedForward(embedding_dim, drop_out)
    self.layer_norm2 = nn.LayerNorm(normalized_shape = embedding_dim)

  def forward(self, x):
    x_block_multihead = self.layer_norm1(self.head(x)) # (batch_size, block_size, embedding_dim)
    x = x + x_block_multihead # (batch_size, block_size, embedding_dim)
    x_block_ffwd = self.layer_norm2(self.ffwd(x)) # (batch_size, block_size, embedding_dim)
    x = x + x_block_ffwd # (batch_size, block_size, embedding_dim)
    return x

class NanoGPT(nn.Module):
  def __init__(self, embedding_dim, block_size, head_size, head_num, drop_out, num_blocks, vocab_size, device):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
    self.pos_emb = nn.Embedding(num_embeddings = block_size, embedding_dim = embedding_dim)
    self.blocks = nn.Sequential(*[Block(embedding_dim, head_size, head_num, drop_out, block_size, device) for _ in range(num_blocks)])
    self.linear = nn.Linear(embedding_dim, vocab_size)
    self.device = device
    self.block_size = block_size

  def forward(self, x):
    # x: (batch_size, block_size)
    x_embd = self.embedding(x) # (batch_size, block_size, embedding_dim)
    pos = torch.arange(start = 0, end = x.shape[1], device = self.device)
    x_pos = self.pos_emb(pos) # (block_size, embedding_dim)
    x = x_embd + x_pos # (batch_size, block_size, embedding_dim)
    x = self.blocks(x) # (batch_size, block_size, embedding_dim)
    x = self.linear(x) # (batch_size, block_size, embedding_dim)
    return x

  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -self.block_size:]
        # get the predictions
        logits = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

  def generate_stream(self, idx, max_new_tokens):
    self.eval()  # set model to evaluation mode
    generated_text = ""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]  # crop to block size
        logits = self(idx_cond)               # (B, T, vocab_size)
        logits = logits[:, -1, :]             # last token logits (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)             # append to context

        # decode the token and print immediately
        token_id = idx_next[0, 0].item()
        token_char = decode([token_id], itos)
        print(token_char, end='', flush=True)
        generated_text += token_char

    print()  # newline at the end
    return generated_text