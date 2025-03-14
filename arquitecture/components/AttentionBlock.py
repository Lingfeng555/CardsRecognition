import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    Attention block that processes a (batch_size, num_features, 7, 7) input
    and returns (batch_size, num_features, attention_value).
    """
    height : int
    width : int
    num_features : int
    attention_value : int
    scale: int
    
    qkv : nn.Linear
    out_proj : nn.Linear
    batch_norm : nn.BatchNorm1d
    
    def __init__(self, num_features, attention_value=1, height: int = 7, width : int = 7):
        super(AttentionBlock, self).__init__()
        
        self.height = height
        self.width = width
        
        self.num_features = num_features
        self.attention_value = attention_value
        self.scale = (height * width) ** -0.5  # Scaling factor

        # Linear projections for Q, K, V
        self.qkv = nn.Linear(height * width, 3 * (height * width), bias=False)
        self.out_proj = nn.Linear(height * width, attention_value)  # Output projection
        self.batch_norm = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x):
        batch_size, num_features, h, w = x.shape  
        assert h == self.height and w == self.width, "Expected input shape [batch_size, num_features, 7, 7]"
        
        x = x.view(batch_size, num_features, -1)  
        
        # Compute Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)  
        q, k, v = qkv

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # Scaled dot-product attention
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_out = torch.matmul(attn_probs, v)

        # Project attention output to the desired shape
        attn_out = self.out_proj(attn_out)

        attn_out = self.batch_norm(attn_out)

        return attn_out