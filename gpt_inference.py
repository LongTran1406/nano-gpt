import torch
from data import encode_data, get_input
from model import NanoGPT, Head, MultiHead, FeedForward, Block

device = "cuda" if torch.cuda.is_available() else "cpu"
user_input = input("Enter your prompt: ")  # e.g., "Hello, how are you?"

chars, text, stoi, itos = get_input()

context_ids = encode_data(user_input, stoi)  # returns a list of token IDs
context = torch.tensor([context_ids], dtype=torch.long, device=device)  # shape: (1, sequence_length)

model = NanoGPT(
    embedding_dim=384,      # use your embedding_dim
    block_size=256,         # use your block_size
    head_size=64,           # your head_size
    head_num=6,             # number of heads
    drop_out=0.2,           # dropout
    num_blocks=3,           # number of transformer blocks
    vocab_size=len(chars),   # vocab size from your data
    device=device
)

model.load_state_dict(torch.load("nanogpt_model_state.pth", map_location=device))
model.to(device)
model.eval()
model.generate_stream(context, max_new_tokens=1000)