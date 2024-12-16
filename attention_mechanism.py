
# attention_mechanism.py
# Created Nov 14th 2024

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = input_size
        self.d_k = input_size // num_heads
        self.d_v = input_size // num_heads

        self.query_layer = nn.Linear(input_size, input_size)
        self.key_layer = nn.Linear(input_size, input_size)
        self.value_layer = nn.Linear(input_size, input_size)

        self.attention_combine = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.size(0)

        # Pass the input through the query, key, and value layers
        queries = self.query_layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        keys = self.key_layer(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        values = self.value_layer(x).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Compute the attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply the attention weights to the values
        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.attention_combine(context)
        output = self.dropout(output)

        return output

class ContextAwareAttention(nn.Module):
    def __init__(self, input_size, context_size, dropout_rate=0.1):
        super(ContextAwareAttention, self).__init__()
        self.input_size = input_size
        self.context_size = context_size

        self.input_projection = nn.Linear(input_size, input_size)
        self.context_projection = nn.Linear(context_size, input_size)
        self.attention_layer = nn.Linear(input_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, context):
        batch_size = x.size(0)

        # Project the input and context vectors
        projected_input = self.input_projection(x)
        projected_context = self.context_projection(context).unsqueeze(1).expand(-1, x.size(1), -1)

        # Compute the attention scores
        attention_scores = self.attention_layer(torch.tanh(projected_input + projected_context)).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply the attention weights to the input
        context_vector = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)
        output = self.dropout(context_vector)

        return output
