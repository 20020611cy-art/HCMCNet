import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoders.VM.VMamba import VSSM, LayerNorm2d, VSSBlock, Permute


class CMDecoder(nn.Module):
    def __init__(self, encoder_dims, norm_layer,channel_first=True,  ssm_act_layer=nn.SiLU, mlp_act_layer=nn.GELU, **kwargs):
        super(CMDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first, ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1)

        # 分类头
        #self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))
        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43)

        p4 = self.fuse_layer_4(
            torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(
            torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(
            torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(
            torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)
        #out = self.main_clf(p1)

        return p1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入VMamba视觉Mamba的核心组件
from models.decoders.VM.VMamba import VSSM, LayerNorm2d, VSSBlock, Permute


class CMDecoder(nn.Module):
    """
    基于VMamba的变化检测解码器 (Change Mamba Decoder)
    核心设计：
    1. 多尺度双时相特征融合：处理编码器输出的4层不同分辨率特征
    2. 三种互补的双时相特征拼接策略：通道拼接、列交错拼接、水平拼接
    3. VSSBlock建模时空长距离依赖：比Transformer更高效的全局上下文建模
    4. 自上而下的渐进式融合：从深层语义到浅层细节的多尺度特征融合
    """

    def __init__(
            self,
            encoder_dims,
            norm_layer,
            channel_first=True,
            ssm_act_layer=nn.SiLU,
            mlp_act_layer=nn.GELU,
            **kwargs
    ):
        super().__init__()

        # ====================== 第4层（最深层，最低分辨率）时空建模模块 ======================
        # st_block_41: 处理通道维度拼接的双时相特征
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1] * 2, out_channels=128),  # 1x1卷积降维到统一128通道
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),  # 非通道优先时转置为(B, H, W, C)
            VSSBlock(  # VMamba核心块，建模时空长距离依赖
                hidden_dim=128,
                drop_path=0.1,
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_act_layer=ssm_act_layer,
                mlp_act_layer=mlp_act_layer
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),  # 转回通道优先格式(B, C, H, W)
        )
        # st_block_42: 处理列交错拼接的双时相特征
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        # st_block_43: 处理水平拼接的双时相特征
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # ====================== 第3层时空建模模块 ======================
        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # ====================== 第2层时空建模模块 ======================
        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # ====================== 第1层（最浅层，最高分辨率）时空建模模块 ======================
        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_act_layer=ssm_act_layer, mlp_act_layer=mlp_act_layer),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # ====================== 特征融合层 ======================
        # 每个层级将5个128通道的特征拼接后降维回128通道
        self.fuse_layer_4 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fuse_layer_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fuse_layer_2 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fuse_layer_1 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # ====================== 平滑层 ======================
        # 残差块消除上采样带来的伪影，平滑融合后的特征
        self.smooth_layer_3 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_2 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.smooth_layer_1 = ResBlock(in_channels=128, out_channels=128, stride=1)

        # 分类头（已注释，可根据任务添加）
        # self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        """FPN风格的上采样加残差融合：将高层特征x上采样到低层特征y的尺寸后相加"""
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, pre_features, post_features):
        """
        前向传播
        Args:
            pre_features: 前时相编码器输出的4层特征 [feat1, feat2, feat3, feat4]
                         feat1: 最高分辨率，最浅层；feat4: 最低分辨率，最深层
            post_features: 后时相编码器输出的4层特征，结构同pre_features
        Returns:
            p1: 最终融合后的变化特征图 (B, 128, H, W)，分辨率与输入图像一致
        """
        # 解包双时相多尺度特征
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features
        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        # ====================== Stage 1: 处理最深层第4层特征 ======================
        # 策略1：通道维度拼接（最常用的双时相融合方式）
        p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))

        B, C, H, W = pre_feat_4.size()
        # 策略2：列交错拼接（奇数列放前时相，偶数列放后时相，模拟逐像素对比）
        ct_tensor_42 = torch.empty(B, C, H, 2 * W, device=pre_feat_4.device)  # 修正：避免硬编码cuda()
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # 奇数列填充前时相特征
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # 偶数列填充后时相特征
        p42 = self.st_block_42(ct_tensor_42)

        # 策略3：水平拼接（左右拼接，保留完整的双时相空间结构）
        ct_tensor_43 = torch.empty(B, C, H, 2 * W, device=pre_feat_4.device)
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4  # 左半部分放前时相
        ct_tensor_43[:, :, :, W:] = post_feat_4  # 右半部分放后时相
        p43 = self.st_block_43(ct_tensor_43)

        # 融合5个特征：p41 + p42拆分的两部分 + p43拆分的两部分
        p4 = self.fuse_layer_4(
            torch.cat([
                p41,
                p42[:, :, :, ::2],  # 提取p42中的前时相部分
                p42[:, :, :, 1::2],  # 提取p42中的后时相部分
                p43[:, :, :, 0:W],  # 提取p43中的前时相部分
                p43[:, :, :, W:]  # 提取p43中的后时相部分
            ], dim=1)
        )

        # ====================== Stage 2: 处理第3层特征 ======================
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))
        B, C, H, W = pre_feat_3.size()

        ct_tensor_32 = torch.empty(B, C, H, 2 * W, device=pre_feat_3.device)
        ct_tensor_32[:, :, :, ::2] = pre_feat_3
        ct_tensor_32[:, :, :, 1::2] = post_feat_3
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2 * W, device=pre_feat_3.device)
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(
            torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1)
        )
        # 自上而下融合：将高层p4上采样后加到当前层p3
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)  # 残差块平滑特征

        # ====================== Stage 3: 处理第2层特征 ======================
        p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))
        B, C, H, W = pre_feat_2.size()

        ct_tensor_22 = torch.empty(B, C, H, 2 * W, device=pre_feat_2.device)
        ct_tensor_22[:, :, :, ::2] = pre_feat_2
        ct_tensor_22[:, :, :, 1::2] = post_feat_2
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2 * W, device=pre_feat_2.device)
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(
            torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1)
        )
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)

        # ====================== Stage 4: 处理最浅层第1层特征 ======================
        p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))
        B, C, H, W = pre_feat_1.size()

        ct_tensor_12 = torch.empty(B, C, H, 2 * W, device=pre_feat_1.device)
        ct_tensor_12[:, :, :, ::2] = pre_feat_1
        ct_tensor_12[:, :, :, 1::2] = post_feat_1
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2 * W, device=pre_feat_1.device)
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(
            torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1)
        )
        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        # 最终分类（已注释）
        # out = self.main_clf(p1)

        return p1


class ResBlock(nn.Module):
    """标准的ResNet残差块，用于特征平滑和消除上采样伪影"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace=True节省内存
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 当输入输出通道或尺寸不同时的下采样层

    def forward(self, x):
        identity = x  # 保存残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要下采样，对残差进行变换
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差相加
        out = self.relu(out)

        return out