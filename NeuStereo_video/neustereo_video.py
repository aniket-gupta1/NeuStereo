import torch
import torch.nn.functional as F

from NeuStereo import backbone
from NeuStereo import transformer
from NeuStereo import matching
from NeuStereo import corr
from NeuStereo import refine
from NeuStereo import upsample
import pdb

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
        # feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def forward(self, img0, img1, warped_prev_disp=None, warped_prev_contexts=None, iters_s16=10, iters_s8=10):
        # print(iters_s16, iters_s8)
        disp_list = []
        img0 /= 255.
        img1 /= 255.
        
        # Extract multi-scale features        
        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))
        
        # Apply Cross Attention and split features
        features_s16 = self.cross_attn_s16(features_s16)
        features_s16, context_s16 = self.split_features(features_s16, self.config.context_dim_s16, self.config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, self.config.context_dim_s8, self.config.feature_dim_s8)
        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)

        if warped_prev_disp is not None:
            # Use the motion-compensated disparity from the previous frame as the initial guess
            # Downsample it to the 1/16th scale
            disp_left = F.interpolate(warped_prev_disp, scale_factor=1./16., mode='bilinear', align_corners=True) / 16.0
        else:
            # For the first frame, compute disparity from scratch
            disp_left = self.matching_s16.stereo_correlation_softmax(feature0_s16, feature1_s16)
        # disp_left = self.matching_s16.stereo_correlation_softmax(feature0_s16, feature1_s16)

        # Initialize the correlation volume
        stereo_corr_pyr_s16 = self.stereo_corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)
        if warped_prev_contexts is not None:
            iter_context_s16 = warped_prev_contexts['s16']
        else:
            iter_context_s16 = self.init_iter_context_s16

        # Refine the 1/16th scale disparity
        for i in range(iters_s16):
            if self.training and i > 0:
                disp_left = disp_left.detach()

            # Compute correlation block and Run refinement network
            stereo_corrs = self.stereo_corr_block_s16(stereo_corr_pyr_s16, disp_left)
            iter_context_s16, stereo_delta_disp = self.stereo_refine_s16(stereo_corrs, context_s16, iter_context_s16, disp_left)

            # Add delta disparity and Upsample for supervision
            disp_left = disp_left + stereo_delta_disp
            if self.training:
                up_disp_left = F.interpolate(disp_left, scale_factor=16, mode='bilinear') * 16
                disp_list.append(up_disp_left)

        # Upsample 1/16th scale features and left image disparity to 1/8th scale
        disp_left = F.interpolate(disp_left, scale_factor=2, mode='nearest') * 2
        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')

        # Merge 1/8th and 1/16th scale features
        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))
        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')
        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))

        # Initialize the correlation volume
        stereo_corr_pyr_s8 = self.stereo_corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)       
        if warped_prev_contexts is not None:
            iter_context_s8 = warped_prev_contexts['s8']
        else:
            iter_context_s8 = self.init_iter_context_s8

        # Refine the 1/8th scale disparity
        for i in range(iters_s8):
            if self.training and i > 0:
                disp_left = disp_left.detach()
            
            # Compute correlation block and Run refinement network
            stereo_corrs = self.stereo_corr_block_s8(stereo_corr_pyr_s8, disp_left)
            iter_context_s8, stereo_delta_disp = self.stereo_refine_s8(stereo_corrs, context_s8, iter_context_s8, disp_left)

            # Add delta disparity and Upsample for supervision
            disp_left = disp_left + stereo_delta_disp
            if self.training or i == iters_s8 - 1:
                feature0_s1 = self.conv_s8(img0)
                up_disp_left = self.upsample_s8(feature0_s1, disp_left) * 8
                disp_list.append(up_disp_left[:, :1])

        final_contexts = {'s16': iter_context_s16, 's8': iter_context_s8}
        return disp_list, final_contexts