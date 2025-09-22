import torch
import torch.nn.functional as F

from NeuStereo_s8_local import backbone
from NeuStereo_s8_local import transformer
from NeuStereo_s8_local import matching
from NeuStereo_s8_local import corr
from NeuStereo_s8_local import refine
from NeuStereo_s8_local import upsample
import pdb

class NeuStereo(torch.nn.Module):
    def __init__(self, config):
        super(NeuStereo, self).__init__()
        self.config = config

        self.backbone = backbone.CNNEncoder(self.config.feature_dim_s16, self.config.context_dim_s16, 
                                            self.config.feature_dim_s8, self.config.context_dim_s8)
        
        self.cross_attn_s8 = transformer.LocalFeatureAttention(self.config.feature_dim_s16+self.config.context_dim_s16, 
                                                               num_layers=2, window_size=self.config.max_disparity // 8)
        
        self.matching_s8 = matching.Matching()

        self.stereo_corr_block_s8 = corr.StereoCorrBlock(radius=4, levels=1)
        
        self.stereo_refine_s8 = refine.StereoRefine(self.config.context_dim_s8, self.config.iter_context_dim_s8, 
                                                    num_layers=5, levels=1, radius=4, inter_dim=96)

        self.conv_s8 = backbone.ConvBlock(3, self.config.feature_dim_s1, kernel_size=8, stride=8, padding=0)
        
        self.upsample_s8 = upsample.UpSample(self.config.feature_dim_s1, upsample_factor=8)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.TIMEIT = False

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height//16, width//16, device, amp)
        self.matching_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.stereo_corr_block_s8.init_bhwd(batch_size, height//8, width//8, device, amp)
        self.stereo_refine_s8.init_bhwd(batch_size, height//8, width//8, device, amp)

        self.init_iter_context_s8 = torch.zeros(batch_size, self.config.iter_context_dim_s8, height//8,
                                                 width//8, device=device, dtype=torch.half if amp else torch.float)

    def split_features(self, features, context_dim, feature_dim):

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        # feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def forward(self, img0, img1, iters_s16=1, iters_s8=8):
        flow_list = []
        timing_dict = {}
        img0 /= 255.
        img1 /= 255.
        
        # Compute backbone features
        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))

        # Compute cross attention to condition features
        features_s8 = self.cross_attn_s8(features_s8)
        
        # Split features in image0 and image1 features
        features_s8, context_s8 = self.split_features(features_s8, self.config.context_dim_s8, self.config.feature_dim_s8)
        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)

        # Match conditioned features to obtain initial disparity prediction
        flow0 = self.matching_s8.stereo_correlation_softmax(feature0_s8, feature1_s8)

        # Refine disparity prediction
        stereo_corr_pyr_s8 = self.stereo_corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)

        iter_context_s8 = self.init_iter_context_s8
        for i in range(iters_s8):
            if self.training and i > 0:
                flow0 = flow0.detach()

            stereo_corrs = self.stereo_corr_block_s8(stereo_corr_pyr_s8, flow0)
            iter_context_s8, stereo_delta_flow = self.stereo_refine_s8(stereo_corrs, context_s8, iter_context_s8, flow0)
            flow0 = flow0 + stereo_delta_flow

            feature0_s1 = self.conv_s8(img0)
            if self.training or i == iters_s8 - 1:
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                flow_list.append(up_flow0[:, :1])
        
        return flow_list