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

    def forward(self, img0, img1, iters_s16=10, iters_s8=10):
        # print(iters_s16, iters_s8)
        flow_list = []
        timing_dict = {}
        img0 /= 255.
        img1 /= 255.
        
        if self.TIMEIT:        
            start_event_total, end_event_total = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event_total.record()
            start_event.record()
        
        features_s16, features_s8 = self.backbone(torch.cat([img0, img1], dim=0))
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Backbone_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s16 = self.cross_attn_s16(features_s16)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Cross_Attn_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s16, context_s16 = self.split_features(features_s16, self.config.context_dim_s16, self.config.feature_dim_s16)
        features_s8, context_s8 = self.split_features(features_s8, self.config.context_dim_s8, self.config.feature_dim_s8)
        feature0_s16, feature1_s16 = features_s16.chunk(chunks=2, dim=0)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Feature_Split_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        flow0 = self.matching_s16.stereo_correlation_softmax(feature0_s16, feature1_s16)
        # print(flow0)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Softmax_corr_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        stereo_corr_pyr_s16 = self.stereo_corr_block_s16.init_corr_pyr(feature0_s16, feature1_s16)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Corr_block_s16_init_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        iter_context_s16 = self.init_iter_context_s16
        for i in range(iters_s16):
            if self.training and i > 0:
                flow0 = flow0.detach()

            stereo_corrs = self.stereo_corr_block_s16(stereo_corr_pyr_s16, flow0)
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s16_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            iter_context_s16, stereo_delta_flow = self.stereo_refine_s16(stereo_corrs, context_s16, iter_context_s16, flow0)
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s16_refine_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            flow0 = flow0 + stereo_delta_flow
            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=16, mode='bilinear') * 16
                flow_list.append(up_flow0)
                if self.TIMEIT:
                    end_event.record()
                    torch.cuda.synchronize()
                    timing_dict[f'Upsample_op_iters_16'] = start_event.elapsed_time(end_event)
                    start_event.record()
    
        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2
        features_s16 = F.interpolate(features_s16, scale_factor=2, mode='nearest')
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Upsample_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s8 = self.merge_s8(torch.cat([features_s8, features_s16], dim=1))
        feature0_s8, feature1_s8 = features_s8.chunk(chunks=2, dim=0)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Merge_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        stereo_corr_pyr_s8 = self.stereo_corr_block_s8.init_corr_pyr(feature0_s8, feature1_s8)
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Corr_block_s8_init_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        context_s16 = F.interpolate(context_s16, scale_factor=2, mode='nearest')
        context_s8 = self.context_merge_s8(torch.cat([context_s8, context_s16], dim=1))
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Context_Merge_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        iter_context_s8 = self.init_iter_context_s8
        for i in range(iters_s8):
            if self.training and i > 0:
                flow0 = flow0.detach()

            stereo_corrs = self.stereo_corr_block_s8(stereo_corr_pyr_s8, flow0)
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s8_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            iter_context_s8, stereo_delta_flow = self.stereo_refine_s8(stereo_corrs, context_s8, iter_context_s8, flow0)
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s8_refine_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            flow0 = flow0 + stereo_delta_flow
            if self.training or i == iters_s8 - 1:
                feature0_s1 = self.conv_s8(img0)
                up_flow0 = self.upsample_s8(feature0_s1, flow0) * 8
                flow_list.append(up_flow0[:, :1])

                if self.TIMEIT:
                    end_event.record()
                    torch.cuda.synchronize()
                    timing_dict[f'Upsample_s8_time'] = start_event.elapsed_time(end_event)
                    start_event.record()

        if self.TIMEIT:
            end_event_total.record()
            torch.cuda.synchronize()
            timing_dict[f'Total_time'] = start_event_total.elapsed_time(end_event_total)
        
        return flow_list