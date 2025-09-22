import torch
import torch.nn.functional as F

from NeuStereo_small import backbone
from NeuStereo_small import transformer
from NeuStereo_small import matching
from NeuStereo_small import corr
from NeuStereo_small import refine
from NeuStereo_small import upsample
import pdb

class NeuStereo(torch.nn.Module):
    def __init__(self, config):
        super(NeuStereo, self).__init__()
        self.config = config

        self.backbone = backbone.CNNEncoder(self.config.feature_dim_s4, self.config.context_dim_s4, self.config.feature_dim_s2, self.config.context_dim_s2)
        
        self.cross_attn_s4 = transformer.FeatureAttention(self.config.feature_dim_s4+self.config.context_dim_s4, num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)
        
        self.matching_s4 = matching.Matching()

        self.stereo_corr_block_s4 = corr.StereoCorrBlock(radius=4, levels=1)
        self.stereo_corr_block_s2 = corr.StereoCorrBlock(radius=4, levels=1)
        
        self.merge_s2 = torch.nn.Sequential(torch.nn.Conv2d(self.config.feature_dim_s4 + self.config.feature_dim_s2, self.config.feature_dim_s2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.GELU(),
                                              torch.nn.Conv2d(self.config.feature_dim_s2, self.config.feature_dim_s2, kernel_size=3, stride=1, padding=1, bias=False),
                                              torch.nn.BatchNorm2d(self.config.feature_dim_s2))

        self.context_merge_s2 = torch.nn.Sequential(torch.nn.Conv2d(self.config.context_dim_s4 + self.config.context_dim_s2, self.config.context_dim_s2, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.GELU(),
                                           torch.nn.Conv2d(self.config.context_dim_s2, self.config.context_dim_s2, kernel_size=3, stride=1, padding=1, bias=False),
                                           torch.nn.BatchNorm2d(self.config.context_dim_s2))

        self.stereo_refine_s4 = refine.StereoRefine(self.config.context_dim_s4, self.config.iter_context_dim_s4, num_layers=5, levels=1, radius=4, inter_dim=128)
        self.stereo_refine_s2 = refine.StereoRefine(self.config.context_dim_s2, self.config.iter_context_dim_s2, num_layers=5, levels=1, radius=4, inter_dim=96)

        self.conv_s2 = backbone.ConvBlock(3, self.config.feature_dim_s1, kernel_size=4, stride=2, padding=1)
        
        self.upsample_s2 = upsample.UpSample(self.config.feature_dim_s1, upsample_factor=2)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.TIMEIT = False

    def init_bhwd(self, batch_size, height, width, device, amp=True):

        self.backbone.init_bhwd(batch_size*2, height, width, device, amp)
        self.matching_s4.init_bhwd(batch_size, height//4, width//4, device, amp)

        self.stereo_corr_block_s4.init_bhwd(batch_size, height//4, width//4, device, amp)
        self.stereo_corr_block_s2.init_bhwd(batch_size, height//2, width//2, device, amp)

        self.stereo_refine_s4.init_bhwd(batch_size, height//4, width//4, device, amp)
        self.stereo_refine_s2.init_bhwd(batch_size, height//2, width//2, device, amp)

        self.init_iter_context_s4 = torch.zeros(batch_size, self.config.iter_context_dim_s4, height//4, width//4, device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s2 = torch.zeros(batch_size, self.config.iter_context_dim_s2, height//2, width//2, device=device, dtype=torch.half if amp else torch.float)

    def split_features(self, features, context_dim, feature_dim):
        context, features = torch.split(features, [context_dim, feature_dim], dim=1)
        context, _ = context.chunk(chunks=2, dim=0)
        return features, torch.relu(context)

    def forward(self, img0, img1, iters_s4=4, iters_s2=10):
        # print(f"s_iter4: {iters_s4}")
        # print(f"s_iter2: {iters_s2}")
        flow_list = []
        timing_dict = {}
        img0 /= 255.
        img1 /= 255.
        # print(f"Input img0 shape: {img0.shape}")
        
        if self.TIMEIT:        
            start_event_total, end_event_total = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event_total.record()
            start_event.record()
        
        features_s4, features_s2 = self.backbone(torch.cat([img0, img1], dim=0))
        # print(f"Backbone output features_s4 shape: {features_s4.shape}")
        # print(f"Backbone output features_s2 shape: {features_s2.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Backbone_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s4 = self.cross_attn_s4(features_s4)
        # print(f"After cross_attn_s4, features_s4 shape: {features_s4.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Cross_Attn_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s4, context_s4 = self.split_features(features_s4, self.config.context_dim_s4, self.config.feature_dim_s4)
        features_s2, context_s2 = self.split_features(features_s2, self.config.context_dim_s2, self.config.feature_dim_s2)
        # print(f"After split, features_s4 shape: {features_s4.shape}, context_s4 shape: {context_s4.shape}")
        # print(f"After split, features_s2 shape: {features_s2.shape}, context_s2 shape: {context_s2.shape}")

        feature0_s4, feature1_s4 = features_s4.chunk(chunks=2, dim=0)
        # print(f"After chunk, feature0_s4 shape: {feature0_s4.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Feature_Split_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        flow0 = self.matching_s4.stereo_correlation_softmax(feature0_s4, feature1_s4)
        # print(f"Initial flow0 from matching_s4 shape: {flow0.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Softmax_corr_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        stereo_corr_pyr_s4 = self.stereo_corr_block_s4.init_corr_pyr(feature0_s4, feature1_s4)
        # print(f"stereo_corr_pyr_s4 shapes: {[c.shape for c in stereo_corr_pyr_s4]}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict['Corr_block_s4_init_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        iter_context_s4 = self.init_iter_context_s4
        for i in range(iters_s4):
            # print(f"Iteration number: {i}")
            if self.training and i > 0:
                flow0 = flow0.detach()

            stereo_corrs = self.stereo_corr_block_s4(stereo_corr_pyr_s4, flow0)
            # if i == 0:
                # print(f"Iter {i} (s4), stereo_corrs shape: {stereo_corrs.shape}")
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s4_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            iter_context_s4, stereo_delta_flow = self.stereo_refine_s4(stereo_corrs, context_s4, iter_context_s4, flow0)
            # if i == 0:
                # print(f"Iter {i} (s4), iter_context_s4 shape: {iter_context_s4.shape}")
                # print(f"Iter {i} (s4), stereo_delta_flow shape: {stereo_delta_flow.shape}")
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s4_refine_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            flow0 = flow0 + stereo_delta_flow
            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=4, mode='bilinear') * 4
                flow_list.append(up_flow0)
                if self.TIMEIT:
                    end_event.record()
                    torch.cuda.synchronize()
                    timing_dict[f'Upsample_op_iters_16'] = start_event.elapsed_time(end_event)
                    start_event.record()
    
        # print(f"After s4 refinement, flow0 shape: {flow0.shape}")
        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2
        features_s4 = F.interpolate(features_s4, scale_factor=2, mode='nearest')
        # print(f"Upsampled to s2, flow0 shape: {flow0.shape}")
        # print(f"Upsampled to s2, features_s4 shape: {features_s4.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Upsample_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        features_s2 = self.merge_s2(torch.cat([features_s2, features_s4], dim=1))
        # print(f"After merge_s2, features_s2 shape: {features_s2.shape}")
        feature0_s2, feature1_s2 = features_s2.chunk(chunks=2, dim=0)
        # print(f"After chunk, feature0_s2 shape: {feature0_s2.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Merge_time'] = start_event.elapsed_time(end_event)
            start_event.record()

        stereo_corr_pyr_s2 = self.stereo_corr_block_s2.init_corr_pyr(feature0_s2, feature1_s2)
        # print(f"stereo_corr_pyr_s2 shapes: {[c.shape for c in stereo_corr_pyr_s2]}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Corr_block_s2_init_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        context_s4 = F.interpolate(context_s4, scale_factor=2, mode='nearest')
        context_s2 = self.context_merge_s2(torch.cat([context_s2, context_s4], dim=1))
        # print(f"After context_merge_s2, context_s2 shape: {context_s2.shape}")
        if self.TIMEIT:
            end_event.record()
            torch.cuda.synchronize()
            timing_dict[f'Context_Merge_time'] = start_event.elapsed_time(end_event)
            start_event.record()
        
        iter_context_s2 = self.init_iter_context_s2
        for i in range(iters_s2):
            if self.training and i > 0:
                flow0 = flow0.detach()

            stereo_corrs = self.stereo_corr_block_s2(stereo_corr_pyr_s2, flow0)
            # if i == 0:
                # print(f"Iter {i} (s2), stereo_corrs shape: {stereo_corrs.shape}")
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s2_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            iter_context_s2, stereo_delta_flow = self.stereo_refine_s2(stereo_corrs, context_s2, iter_context_s2, flow0)
            # if i == 0:
                # print(f"Iter {i} (s2), iter_context_s2 shape: {iter_context_s2.shape}")
                # print(f"Iter {i} (s2), stereo_delta_flow shape: {stereo_delta_flow.shape}")
            if self.TIMEIT:
                end_event.record()
                torch.cuda.synchronize()
                timing_dict[f'Corr_block_s2_refine_iter{i}'] = start_event.elapsed_time(end_event)
                start_event.record()

            flow0 = flow0 + stereo_delta_flow
            if self.training or i == iters_s2 - 1:
                feature0_s1 = self.conv_s2(img0)
                up_flow0 = self.upsample_s2(feature0_s1, flow0) * 2
                # if i == iters_s2 - 1:
                    # print(f"Final upsampled flow shape: {up_flow0.shape}")
                flow_list.append(up_flow0[:, :1])

                if self.TIMEIT:
                    end_event.record()
                    torch.cuda.synchronize()
                    timing_dict[f'Upsample_s2_time'] = start_event.elapsed_time(end_event)
                    start_event.record()

        if self.TIMEIT:
            end_event_total.record()
            torch.cuda.synchronize()
            timing_dict[f'Total_time'] = start_event_total.elapsed_time(end_event_total)
        
        return flow_list