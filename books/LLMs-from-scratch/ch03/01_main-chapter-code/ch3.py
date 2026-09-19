
import torch
import torch.nn as nn

inputs = torch.tensor([[0.43, 0.15, 0.89],  # Your
                           [0.55, 0.87, 0.66],  # journey
                           [0.57, 0.85, 0.64],  # starts
                           [0.22, 0.58, 0.33],  # with
                           [0.77, 0.25, 0.10],  # one
                           [0.05, 0.80, 0.55]]  # step
                          )

def part_3_1():


    print(inputs)

    query = inputs[1]
    print(query)
    print(inputs.size())
    attn_scores_2 = torch.empty(inputs.shape[0])
    print(attn_scores_2)

    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(query, x_i)
    print(attn_scores_2)

    # attn_scores_2_trial = torch.empty(inputs.shape[0])
    attn_scores_2_trial = inputs @ inputs.T
    print(attn_scores_2_trial)

    # attn_weigts = attn_scores_2_trial / attn_scores_2_trial.sum(dim=1, keepdim=True)
    attn_weights = torch.softmax(attn_scores_2_trial, dim=1)
    print(attn_weights)

    context_vecs = attn_weights @ inputs
    print(context_vecs)

def part_3_2():

    d_in = inputs.shape[1]
    d_out = 2

    torch.manual_seed(123)
    w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query = inputs @ w_query
    print(f"w_query: {query}")
    key = inputs @ w_key
    value = inputs @ w_value
    attn_scores = query @ key.T
    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    attn_weights = torch.softmax(masked / key.shape[-1]**0.5, dim=-1)
    context_vecs = attn_weights @ value
    print(context_vecs)

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_hid, qkv_bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_hid, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_hid, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_hid, bias=qkv_bias)

    def forward(self, x):
        queries = self.w_query(x)
        keys =  self.w_key(x)
        values = self.w_value(x)

        attn_scores = queries @ keys.T
        context_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
        print(attn_weights)
        context_vecs = attn_weights @ values
        return context_vecs



if __name__ == "__main__":
    # part_3_1()
    # part_3_2()
    torch.manual_seed(789)
    sa = SelfAttention(inputs.shape[1],2)
    print(sa(inputs))
