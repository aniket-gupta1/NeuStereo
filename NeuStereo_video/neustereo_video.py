import torch
import torch.nn.functional as F

from NeuStereo_video import backbone
from NeuStereo_video import transformer
from NeuStereo_video import matching
from NeuStereo_video import corr
from NeuStereo_video import refine
from NeuStereo_video import upsample
import pdb

def check_tensor(tensor, name):
    """Checks for nan or inf in a tensor and prints a debug message."""
    if torch.isnan(tensor).any():
        print(f"DEBUG: 🔴 NaN found in '{name}'")
    elif torch.isinf(tensor).any():
        print(f"DEBUG: 🔴 Inf found in '{name}'")

class NeuStereo(torch.nn.Module):
    def __init__(self, config):
        super(NeuStereo, self).__init__()
        self.config = config

        self.backbone = backbone.CNNEncoder(self.config.feature_dim_s16, self.config.context_dim_s16, self.config.feature_dim_s8, self.config.context_dim_s8)
        
        self.cross_attn_s16 = transformer.FeatureAttention(self.config.feature_dim_s16+self.config.context_dim_s16, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s16 = matching.Matching()

        self.stereo_corr_block_s16 = corr.StereoCorrBlock(radius=4, levels=1)
        self.stereo_corr_block_s8 = corr.StereoCorrBlock(radius=4, levels=1)
        
        self.merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(self.config.feature_dim_s16 + self.config.feature_dim_s8, self.config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(self.config.feature_dim_s8, self.config.feature_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.BatchNorm2d(self.config.feature_dim_s8))

        self.context_merge_s8 = torch.nn.Sequential(torch.nn.Conv2d(self.config.context_dim_s16 + self.config.context_dim_s8, self.config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(self.config.context_dim_s8, self.config.context_dim_s8, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(self.config.context_dim_s8))

        self.stereo_refine_s16 = refine.StereoRefine(self.config.context_dim_s16, self.config.iter_context_dim_s16, num_layers=5, levels=1, radius=4, inter_dim=128)
        self.stereo_refine_s8 = refine.StereoRefine(self.config.context_dim_s8, self.config.iter_context_dim_s8, num_layers=5, levels=1, radius=4, inter_dim=96)

        self.conv_s8 = backbone.ConvBlock(3, self.config.feature_dim_s1, kernel_size=8, stride=8, padding=0)
        
        self.upsample_s8 = upsample.UpSample(self.config.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.TIMEIT = False

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height//16, width//16, device, amp)
        self.matching_s16.init_bhwd(batch_size, height//16, width//16, device, amp)

        self.stereo_corr_block_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.stereo_corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.stereo_refine_s16.init_bhwd(batch_size, height//16, width//16, device, amp)
        self.stereo_refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.init_iter_context_s16 = torch.zeros(batch_size, self.config.iter_context_dim_s16, height//16, width//16, device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s8 = torch.zeros(batch_size, self.config.iter_context_dim_s8, height//8, width//8, device=device, dtype=torch.half if amp else torch.float)

    def split_features(self, features, context_dim, feature_dim):

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        return features, torch.relu(context)

    def forward(self, img0, img1, warped_prev_disp=None, warped_prev_contexts=None, iters_s16=10, iters_s8=10):
        disp_list = []
        img0 /= 255.
        img1 /= 255.

        # --- DEBUG: Check inputs ---
        check_tensor(img0, "Input img0")
        check_tensor(img1, "Input img1")

        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))
        check_tensor(features_s16, "Backbone features_s16")
        check_tensor(features_s8, "Backbone features_s8")
        
        features_s16 = self.cross_attn_s16(features_s16)
        check_tensor(features_s16, "Cross-Attention s16")

        features_s16, context_s16 = self.split_features(features_s16, self.config.context_dim_s16, self.config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, self.config.context_dim_s8, self.config.feature_dim_s8)
        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        if warped_prev_disp is not None:
            disp_left = F.interpolate(warped_prev_disp, scale_factor=1./16., mode='bilinear', align_corners=True) / 16.0
        else:
            disp_left = self.matching_s16.stereo_correlation_softmax(feature0_s16, feature1_s16)
        check_tensor(disp_left, "Initial Disparity (s16)")

        stereo_corr_pyr_s16 = self.stereo_corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)
        iter_context_s16 = self.init_iter_context_s16 if warped_prev_contexts is None else warped_prev_contexts['s16']

        for i in range(iters_s16):
            if self.training and i > 0:
                disp_left = disp_left.detach()

            stereo_corrs = self.stereo_corr_block_s16(stereo_corr_pyr_s16, disp_left)
            check_tensor(stereo_corrs, f"s16 Refine Iter {i} - Input Corrs")
            
            iter_context_s16, stereo_delta_disp = self.stereo_refine_s16(stereo_corrs, context_s16, iter_context_s16, disp_left)
            check_tensor(stereo_delta_disp, f"s16 Refine Iter {i} - Delta Disp")

            disp_left = disp_left + stereo_delta_disp
            disp_left = torch.clamp(disp_left, min=0, max=self.config.max_disp / 16.0) # Keep the clamp just in case
            check_tensor(disp_left, f"s16 Refine Iter {i} - Updated Disp")

            if self.training:
                disp_list.append(F.interpolate(disp_left, scale_factor=16, mode='bilinear') * 16)

        disp_left = F.interpolate(disp_left, scale_factor=2, mode='nearest') * 2
        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))
        check_tensor(features_s8, "Merged Features (s8)")
        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')
        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        stereo_corr_pyr_s8 = self.stereo_corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)       
        iter_context_s8 = self.init_iter_context_s8 if warped_prev_contexts is None else warped_prev_contexts['s8']

        for i in range(iters_s8):
            if self.training and i > 0:
                disp_left = disp_left.detach()
            
            stereo_corrs = self.stereo_corr_block_s8(stereo_corr_pyr_s8, disp_left)
            check_tensor(stereo_corrs, f"s8 Refine Iter {i} - Input Corrs")

            iter_context_s8, stereo_delta_disp = self.stereo_refine_s8(stereo_corrs, context_s8, iter_context_s8, disp_left)
            check_tensor(stereo_delta_disp, f"s8 Refine Iter {i} - Delta Disp")

            disp_left = disp_left + stereo_delta_disp
            disp_left = torch.clamp(disp_left, min=0, max=self.config.max_disp / 8.0)
            check_tensor(disp_left, f"s8 Refine Iter {i} - Updated Disp")
            
            if self.training or i == iters_s8 - 1:
                feature0_s1 = self.conv_s8(img0)
                up_disp_left = self.upsample_s8(feature0_s1, disp_left) * 8
                disp_list.append(up_disp_left[:, :1])

        final_contexts = {'s16': iter_context_s16, 's8': iter_context_s8}
        return disp_list, final_contexts