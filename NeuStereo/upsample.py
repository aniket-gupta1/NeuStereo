import torch
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler
import pdb

class UpSample(torch.nn.Module):
    def __init__(self, feature_dim, upsample_factor):
        super(UpSample, self).__init__()

        self.upsample_factor = upsample_factor

        self.conv1 = torch.nn.Conv2d(1 + feature_dim, 256, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(512, upsample_factor ** 2 * 9, 1, 1, 0)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, feature, flow):
        pdb.set_trace()

        concat = torch.cat((flow, feature), dim=1)

        mask = self.conv3(self.relu(self.conv2(self.relu(self.conv1(concat)))))

        b, _, h, w = flow.shape

        mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        # up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(b, 1, 9, 1, 1, h, w)  # [B, 1, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 1, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 1, K, H, K, W]
        up_flow = up_flow.reshape(b, 1, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 1, K*H, K*W]

        return up_flow

class LayeredResizeConv(torch.nn.Module):
    def __init__(self, dim, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv2 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        self.conv3 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
        # self.conv4 = torch.nn.Conv2d(dim + 3, dim, kernel_size, padding="same")
 
    def apply_conv(self, source, guidance, conv, activation):
        big_source = F.interpolate(source, scale_factor=2, mode="bilinear")
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = activation(conv(torch.cat([big_source, small_guidance], dim=1)))
        return big_source + output
 
    def forward(self, source, guidance):
        """
        source: initial optical flow or stereo disparity [B, 1 or 2, H//8, W//8]
        guidance: original image guidance [B, 3, H, W]
            Note: We can also use features from the encoder as guidance.

        At each step here, we upsample the source by a factor of 2 and apply a convolutional layer.

        output: upsampled optical flow or stereo disparity [B, 1 or 2, H, W]
        """
        source_2 = self.apply_conv(source, guidance, self.conv1, F.relu)
        source_4 = self.apply_conv(source_2, guidance, self.conv2, F.relu)
        source_8 = self.apply_conv(source_4, guidance, self.conv3, lambda x: x)
        # source_16 = self.apply_conv(source_8, guidance, self.conv4, lambda x: x)
        return source_8