import data as data
import torch
import torch.nn as nn

class GPT(nn.Module):

    def __init__(self, vocab_size, dim, max_t):
        super().__init__() 
        self.max_t = max_t
        self.token_table = nn.Embedding(vocab_size, dim)
        self.pos_table = nn.Embedding(max_t, dim)


    def forward(self, ids):
        token_vec = self.token_table(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        pos_vec = self.pos_table(pos)
        fin_vec = token_vec + pos_vec
        return fin_vec


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = GPT(len(data.chars), 384, 256).to(device)
    x, y = data.batch(32, 256, data.train_set)
    out = model(torch.from_numpy(x).to(device))
    print(out.shape)
