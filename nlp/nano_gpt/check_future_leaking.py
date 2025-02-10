import torch
from nlp.nano_gpt.model import BatchWhiteningBlock
import torch.nn as nn

batch, sentence_length, embedding_dim = 1, 1024, 768

embedding = torch.randn(batch, sentence_length, embedding_dim,requires_grad=True)

#layer_norm = BatchWhiteningBlock(embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)

# Activate module

output = layer_norm(embedding)



grad = torch.autograd.grad(output[:, 3, :].sum(), embedding)[0]

print('the gradient of output[3] w.r.t token 4,3,2 : should be 0 for token2 and token4), and !=0 for token 3')

print(grad[0, 4, :].sum(), grad[0, 3, :].sum(),grad[0, 2, :].sum())