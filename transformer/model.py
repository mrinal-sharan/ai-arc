import data as data
import torch
import torch.nn as nn


# class Head(nn.Module):

#     def __init__(self, C, head_size, max_t):
#         super().__init__()
#         self.W_q = nn.Linear(C, head_size, bias=False)
#         self.W_k = nn.Linear(C, head_size, bias=False)
#         self.W_v = nn.Linear(C, head_size, bias=False)


#         self.register_buffer("mask", torch.tril(torch.ones(max_t, max_t)))
#         self.identity = nn.Identity()



#     def forward(self, x):
#         q = self.W_q(x) 
#         k = self.W_k(x) 
#         v = self.W_v(x)
#         logits = (q @ k.mT)/q.shape[-1]**0.5
#         logits = self.identity(logits)
#         T = logits.shape[-1]
#         logits = logits.masked_fill(self.mask[:T,:T] ==0, float('-inf'))
#         weights = torch.softmax(logits, dim=-1)

#         return weights @ v


class GPT(nn.Module):

    def __init__(self, vocab_size, dim, max_t, head_size, n_layers):
        super().__init__() 
        self.max_t = max_t
        self.token_table = nn.Embedding(vocab_size, dim)
        self.pos_table = nn.Embedding(max_t, dim)
        self.blocks = nn.ModuleList([ Block(dim, max_t, head_size) for _ in range(n_layers)])


    def forward(self, ids):
        token_vec = self.token_table(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        pos_vec = self.pos_table(pos)
        fin_vec = token_vec + pos_vec

        for layer in self.blocks:
            fin_vec = layer(fin_vec)

        return fin_vec


class MultiHeadAttention(nn.Module):

    def __init__(self, max_t, dim, head_size):
        super().__init__() 
        self.head_size = head_size
        self.dim = dim
        self.multi_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.merge = nn.Linear(dim, dim, bias =False)
        self.register_buffer("mask", torch.tril(torch.ones(max_t, max_t)))
        self.identity = nn.Identity()


    def forward(self, x):

        multi_head_mat = self.multi_qkv(x)
        q, k , v = multi_head_mat.split(self.dim, dim=-1)
        heads = (multi_head_mat.shape[-1]//3) // self.head_size
        q = q.view(x.shape[0], x.shape[1], heads, self.head_size)
        k = k.view(x.shape[0], x.shape[1], heads, self.head_size)
        v = v.view(x.shape[0], x.shape[1], heads, self.head_size)

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        logits = (q @ k.mT)/q.shape[-1]**0.5

        logits = self.identity(logits)
        T = logits.shape[-1]
        logits = logits.masked_fill(self.mask[:T,:T] ==0, float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        res = weights @ v

        res = res.transpose(-3, -2)
        res = res.reshape(x.shape[0], x.shape[1], self.dim)
        return self.merge(res)


class FeedForward(nn.Module):
    def __init__(self, C):
        super().__init__() 
        self.Feed_1= nn.Linear(C, 4*C, bias =False)
        self.Feed_2 = nn.Linear(4*C, C, bias =False)
        self.GELU = nn.GELU()

    def forward(self, x):
        x = self.Feed_1(x)
        x = self.GELU(x)
        x = self.Feed_2(x)
        return x

class Block(nn.Module):
    def __init__(self, C, max_t, head_size):
        super().__init__() 
        self.head_size = head_size
        self.normal = nn.LayerNorm(C, bias =False)
        self.normal_2 = nn.LayerNorm(C, bias =False)
        self.Multihead = MultiHeadAttention(max_t, C, head_size)
        self.FFN = FeedForward(C)
        

    def forward(self,x):
        
        x = x + self.Multihead(self.normal(x))
        x = x + self.FFN(self.normal_2(x))
       


        return x
        


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = GPT(len(data.chars), 384, 256, 64, 6).to(device)
    x, y = data.batch(32, 256, data.train_set)
    out = model(torch.from_numpy(x).to(device))
    print(out.shape)
