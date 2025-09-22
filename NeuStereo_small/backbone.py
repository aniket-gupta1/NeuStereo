import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A lightweight convolutional block using two convolutions.
    The first convolution can be strided to perform downsampling.
    Using LeakyReLU and BatchNorm for stability.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        return x


class CNNEncoder(torch.nn.Module):
    """
    A shallow, lightweight CNN encoder that extracts features AND context
    at both 1/2 and 1/4 resolutions, maintaining a parallel design.
    """
    def __init__(self, f_dim_2=32, c_dim_2=16, f_dim_4=64, c_dim_4=32):
        super(CNNEncoder, self).__init__()

        # --- Blocks for 1/2 Scale ---
        # Path 1: A shallow path from the original image to 1/2 scale.
        self.block_s2_path1 = ConvBlock(3, f_dim_2, kernel_size=4, stride=2, padding=1)
        # Path 2: Another shallow path, using a larger kernel to capture different features.
        self.block_s2_path2 = ConvBlock(3, f_dim_2, kernel_size=8, stride=2, padding=3)
        # Merge Block: Combines the two parallel 1/2 scale paths to create features and context.
        self.block_cat_s2 = ConvBlock(f_dim_2 * 2, f_dim_2 + c_dim_2, kernel_size=3, stride=1, padding=1)

        # --- Blocks for 1/4 Scale ---
        # Path 1: Processes a 1/4 scale version of the input image.
        self.block_s4_path1 = ConvBlock(3, f_dim_4, kernel_size=3, stride=1, padding=1)
        # Path 2: Downsamples the merged 1/2 scale features to 1/4 scale.
        self.block_s2_to_s4 = ConvBlock(f_dim_2 + c_dim_2, f_dim_4, kernel_size=4, stride=2, padding=1)
        # Merge Block: Combines the two parallel 1/4 scale paths.
        self.block_cat_s4 = ConvBlock(f_dim_4 * 2, f_dim_4 + c_dim_4 - 2, kernel_size=3, stride=1, padding=1)

        self.pos_s4 = None
        self.pos_s2 = None

    def _init_pos(self, batch_size, height, width, device, amp):
        """Helper to create normalized positional encoding."""
        dtype = torch.half if amp else torch.float
        ys, xs = torch.meshgrid(torch.arange(height, dtype=dtype, device=device),
                                torch.arange(width, dtype=dtype, device=device), indexing='ij')
        ys_norm = (ys - (height - 1) / 2.0) / ((height - 1) / 2.0)
        xs_norm = (xs - (width - 1) / 2.0) / ((width - 1) / 2.0)
        pos = torch.stack([ys_norm, xs_norm], dim=0)
        return pos[None].repeat(batch_size, 1, 1, 1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        """Initializes positional encodings for the feature scales."""
        self.pos_s2 = self._init_pos(batch_size, height // 2, width // 2, device, amp)
        self.pos_s4 = self._init_pos(batch_size, height // 4, width // 4, device, amp)

    def forward(self, img):
        # print(f"Input img shape: {img.shape}")
        # --- 1/2 Scale Feature and Context Extraction ---
        # Run two parallel conv blocks on the input image. 
        s2_p1 = self.block_s2_path1(img)
        s2_p2 = self.block_s2_path2(img)
        # print(f"s2_p1 shape: {s2_p1.shape}, s2_p2 shape: {s2_p2.shape}")
        # Merge the parallel streams to get the final 1/2 scale features + context.
        features_s2_merged = self.block_cat_s2(torch.cat([s2_p1, s2_p2], dim=1))
        # print(f"features_s2_merged shape: {features_s2_merged.shape}")

        # --- 1/4 Scale Feature and Context Extraction ---
        # Create a downsampled 1/4 image for the first path.
        img_s4 = F.avg_pool2d(img, kernel_size=4, stride=4)
        # print(f"img_s4 shape: {img_s4.shape}")
        # Path 1: Process the downsampled image.
        s4_p1 = self.block_s4_path1(img_s4)
        # Path 2: Downsample the merged 1/2 scale features.
        s4_p2 = self.block_s2_to_s4(features_s2_merged)
        # print(f"s4_p1 shape: {s4_p1.shape}, s4_p2 shape: {s4_p2.shape}")
        # Merge the parallel streams to get the final 1/4 scale features + context.
        features_s4_merged = self.block_cat_s4(torch.cat([s4_p1, s4_p2], dim=1))
        # print(f"features_s4_merged shape: {features_s4_merged.shape}")

        # --- Add Positional Encoding ---
        # features_s2_pos = torch.cat([features_s2_merged, self.pos_s2], dim=1)
        # print(f"self.pos_s4; {self.pos_s4.shape}")
        features_s4_pos = torch.cat([features_s4_merged, self.pos_s4], dim=1)
        # print(f"features_s4_pos (with positional encoding) shape: {features_s4_pos.shape}")

        # Return features from coarsest (1/4) to finest (1/2).
        return features_s4_pos, features_s2_merged
