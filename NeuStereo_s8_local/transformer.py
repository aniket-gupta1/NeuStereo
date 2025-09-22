import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 ffn=True,
                 ffn_dim_expansion=1
                 ):
        super(TransformerLayer, self).__init__()

        # multi-head attention
        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        # self.multi_head_attn = torch.nn.MultiheadAttention(feature_dim, 2, batch_first=True, device='cuda')

        self.norm1 = torch.nn.LayerNorm(feature_dim)

        self.ffn = ffn

        if self.ffn:
            in_channels = feature_dim * 2
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
            )

            self.norm2 = torch.nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        message = self.merge(message)

        # message, _ = self.multi_head_attn(query, key, value, need_weights=False)
        message = self.norm1(message)

        if self.ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

class FeatureAttention(torch.nn.Module):
    def __init__(self, feature_dim, num_layers, ffn=True, ffn_dim_expansion=1, post_norm=False):
        super(FeatureAttention, self).__init__()

        self.layers = torch.nn.ModuleList([
            TransformerLayer(feature_dim, ffn=ffn, ffn_dim_expansion=ffn_dim_expansion
                             )
            for i in range(num_layers)])

        self.post_norm = post_norm

        if self.post_norm:
            self.norm = torch.nn.BatchNorm2d(feature_dim)

    def forward(self, concat_features0):

        # 2D Attention ->
        b, c, h, w = concat_features0.shape

        concat_features0 = concat_features0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)
        for layer in self.layers:
            concat_features0 = layer(concat_features0, concat_features1)
            concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # reshape back
        concat_features0 = concat_features0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        if self.post_norm:
            concat_features0 = self.norm(concat_features0)


        # 1D Attention ->
        # B,C,H,W = concat_features0.shape
        # concat_features0 = concat_features0.permute(0, 2, 1, 3).reshape(B * H, W, C)  # [B*H, W, C]
        # concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)
        # # pdb.set_trace()
        # # assert concat_features0.shape == (B*H, W, C) and concat_features1.shape == (B*H, W, C)

        # for layer in self.layers:
        #     concat_features0 = layer(concat_features0, concat_features1)
        #     concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # # reshape back
        # concat_features0 = concat_features0.view(B,H,W,C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # if self.post_norm:
        #     concat_features0 = self.norm(concat_features0)

        return concat_features0

class FlowAttention(torch.nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, feature_dim):
        super(FlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature.size()

        feature = feature.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        flow = flow.flatten(-2).permute(0, 2, 1)

        query = self.q_proj(feature)  # [B, H*W, C]
        key = self.k_proj(feature)  # [B, H*W, C]

        flow = F.scaled_dot_product_attention(query, key, flow)

        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2)

        return flow


# You need to install this library first:
# pip install local-attention
from local_attention import LocalAttention

class LocalTransformerLayer(nn.Module):
    """
    Transformer layer with local cross-attention.
    Replaces the global attention with an efficient, windowed version.
    """
    def __init__(self,
                 feature_dim,
                 ffn_dim_expansion=1,
                 window_size=128 # Corresponds to max disparity
                 ):
        super(LocalTransformerLayer, self).__init__()

        # Local attention module from lucidrains' library
        self.attn = LocalAttention(
            dim=feature_dim,
            window_size=window_size,
            look_backward=True,  # Crucial for stereo: attend to pixels on the left
            look_forward=False,  # Do not attend to pixels on the right
            causal=False,        # Not a sequential/autoregressive task
            autopad=True,        # Pads the input to be divisible by window size
            exact_windowsize=False # Uses a simpler windowing logic
        )

        self.norm1 = nn.LayerNorm(feature_dim)

        # Feed-forward network (optional but common)
        in_channels = feature_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
            nn.GELU(),
            nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
        )

        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B*H, W, C]
        
        # The LocalAttention module takes q, k, v as input
        # For cross-attention, query is from source, key/value are from target
        message = self.attn(source, target, target)

        # Residual connection and normalization
        source_with_message = source + self.norm1(message)
        
        # Feed-forward network step
        message = self.mlp(torch.cat([source_with_message, message], dim=-1))
        message = self.norm2(message)
        
        return source_with_message + message


class LocalFeatureAttention(nn.Module):
    """
    Applies local cross-attention between left and right image features.
    """
    def __init__(self, feature_dim, num_layers, window_size, ffn_dim_expansion=1):
        super(LocalFeatureAttention, self).__init__()

        self.layers = nn.ModuleList([
            LocalTransformerLayer(feature_dim=feature_dim,
                                  ffn_dim_expansion=ffn_dim_expansion,
                                  window_size=window_size)
            for _ in range(num_layers)])

    def forward(self, concat_features):
        # concat_features shape: [B_doubled, C, H, W]
        # where B_doubled is 2 * actual_batch_size (for left and right features)
        B_doubled, C, H, W = concat_features.shape
        B = B_doubled // 2

        # Split concatenated features into left and right features
        features0, features1 = concat_features.chunk(chunks=2, dim=0)

        # Reshape for row-wise local attention: [B, C, H, W] -> [B*H, W, C]
        # This treats each row as an independent sequence for the attention mechanism.
        features0 = features0.permute(0, 2, 3, 1).reshape(B * H, W, C)
        features1 = features1.permute(0, 2, 3, 1).reshape(B * H, W, C)

        # Apply symmetric cross-attention
        for layer in self.layers:
            # Attend from left to right, and right to left
            features0_updated = layer(features0, features1)
            features1_updated = layer(features1, features0)
            
            # Update features for the next layer
            features0, features1 = features0_updated, features1_updated

        # Reshape back to original image format: [B*H, W, C] -> [B, C, H, W]
        features0 = features0.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        features1 = features1.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # Concatenate back into a single tensor
        return torch.cat([features0, features1], dim=0)

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    batch_size = 2
    feature_dim = 128  # C
    height = 32        # H
    width = 64         # W
    max_disparity = 48 # This will be our window size

    # Create dummy left and right feature maps
    left_features = torch.randn(batch_size, feature_dim, height, width).cuda()
    right_features = torch.randn(batch_size, feature_dim, height, width).cuda()

    # Concatenate them along the batch dimension, as in your original code
    concatenated_features = torch.cat([left_features, right_features], dim=0)

    # Instantiate the local attention module
    local_attention_module = LocalFeatureAttention(
        feature_dim=feature_dim,
        num_layers=4,
        window_size=max_disparity
    ).cuda()

    # Process the features
    output_features = local_attention_module(concatenated_features)

    print("Input shape:", concatenated_features.shape)
    print("Output shape:", output_features.shape)
    
    # Verify the output shape is correct
    assert concatenated_features.shape == output_features.shape
    print("\n✅ Local attention module executed successfully!")