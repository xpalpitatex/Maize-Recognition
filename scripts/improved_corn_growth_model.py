import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import timm
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
import cv2
from PIL import Image
import io
import pathlib

# 生育期类别定义
GROWTH_STAGES = {
    '11': '播种期',
    '21': '出苗期',
    '31': '三叶期',
    '41': '七叶期',
    '61': '拔节期',
    '71': '抽雄期',
    '81': '乳熟期',
    '91': '成熟期',
    '99': '收割后'
}

# 模型配置 - 时间特征消融实验版本 (保留图像增强组件)
CONFIG = {
    'img_size': 512,             # 标准图像尺寸
    'batch_size': 16,            # 标准批次大小
    'epochs': 200,               # 训练周期
    'patience': 100,              # 早停耐心
    'base_lr': 5e-5,             # 保守的学习率
    'min_lr': 1e-7,              # 最小学习率
    'dropout_rate': 0.3,         # 适中的Dropout
    'weight_decay': 1e-4,        # 适中的L2正则化
    'num_classes': len(GROWTH_STAGES),  # 类别数量
    'train_ratio': 0.7,          # 训练集比例
    'val_ratio': 0.15,           # 验证集比例
    'test_ratio': 0.15,          # 测试集比例
    'n_splits': 5,               # 交叉验证折数
    'base_model': 'resnet50',    # 使用标准模型
    'use_temporal_info': False,   # ❌ 消融实验：禁用时间信息
    'use_attention': True,       # ✅ 保留注意力机制 (图像相关)
    'use_mixup': False,          # 禁用MixUp数据增强
    'use_cutmix': False,         # 禁用CutMix数据增强
    'use_focal_loss': False,     # 使用简单交叉熵
    'gamma': 1.0,                # Focal Loss的gamma参数
    'use_balanced_sampling': True, # 使用平衡采样策略
    'use_time_consistency': False, # ❌ 禁用时间一致性约束 (时间相关)
    'use_progressive_resizing': False, # 关闭渐进式尺寸调整
    'class_weights_power': 0.5,   # 类权重计算的幂参数
    'use_multiscale_features': True,  # ✅ 保留多尺度特征提取 (图像相关)
    'use_difficult_stage_enhancement': True,  # ✅ 保留困难样本增强 (图像相关)
    'use_contrastive_learning': False,  # 关闭对比学习 (简化)
    'contrastive_weight': 0.0,    # 对比学习损失权重

     
    'label_smoothing': 0.0,       # 禁用标签平滑
    'gradient_clip': 2.0,         # 增加梯度裁剪
    'freeze_backbone_epochs': 0, # 不冻结backbone
    'use_stochastic_depth': False, # 禁用随机深度
    'stochastic_depth_prob': 0.0  # 随机深度概率
}

class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器 - 捕获不同尺度的细节，专门优化小目标检测"""
    def __init__(self, in_channels=3):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # 原有的3个尺度 (保持兼容性)
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 新增：小目标专用尺度
        self.micro_scale = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0),  # 1x1 捕获像素级细节
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_scale = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=2, dilation=2),  # 膨胀卷积扩大感受野
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 更新融合层以处理5个尺度 (3*64 + 2*32 = 256)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 原有的3个尺度
        s1 = self.scale1(x)  # 3x3 局部纹理
        s2 = self.scale2(x)  # 5x5 小目标轮廓
        s3 = self.scale3(x)  # 7x7 上下文信息
        
        # 新增的小目标专用尺度
        micro = self.micro_scale(x)     # 1x1 像素级细节，专门捕获幼苗颜色变化
        dilated = self.dilated_scale(x) # 膨胀卷积，高效扩大感受野
        
        # 拼接所有尺度特征
        fused = torch.cat([s1, s2, s3, micro, dilated], dim=1)
        output = self.fusion(fused)
        
        return output

class TimeFeatureModule(nn.Module):
    """时间特征处理模块 - 简化稳定版本"""
    
    def __init__(self, time_dim=3, hidden_dim=128, dropout_rate=0.3, 
                 noise_std=0.0, use_noise=False, time_importance=1.0):  # 禁用噪声，简化权重
        """
        Args:
            time_dim: 时间特征维度 (年内时间, 时间正弦, 时间余弦)
            hidden_dim: 隐藏层维度
            dropout_rate: Dropout比率
            noise_std: 噪声标准差 (设为0禁用)
            use_noise: 是否在训练时添加噪声 (设为False禁用)
            time_importance: 时间特征的重要性权重 (设为1.0禁用缩放)
        """
        super(TimeFeatureModule, self).__init__()
        
        self.noise_std = 0.0  # 强制禁用噪声
        self.use_noise = False  # 强制禁用噪声
        
        # 简化的时间编码器 - 只用一层
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm提高稳定性
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # 移除复杂的权重控制，直接使用固定权重
        # self.time_importance = nn.Parameter(torch.tensor(time_importance))
        # self.time_dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 输入x: [batch_size, time_dim] (3维)
        
        # 检查输入的有效性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: 时间特征输入包含NaN或Inf")
            x = torch.clamp(x, -10.0, 10.0)  # 限制数值范围
        
        # 简化的时间特征编码 - 移除噪声和复杂操作
        x = self.time_encoder(x)
        
        # 检查输出的有效性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: 时间特征编码输出包含NaN或Inf")
            x = torch.zeros_like(x)  # 如果出现异常，返回零张量
        
        return x

class ChannelAttention(nn.Module):
    """增强版通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """增强版空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)

class VegetationIndexAttention(nn.Module):
    """植被指数注意力 - 基于NDVI思想增强绿色植被特征"""
    def __init__(self, in_channels):
        super(VegetationIndexAttention, self).__init__()
        
        # 学习RGB到植被指数的映射
        self.vegetation_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 绿色增强分支
        self.green_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 计算植被指数权重
        veg_weight = self.vegetation_conv(x)
        
        # 绿色特征增强
        green_enhanced = self.green_enhancer(x)
        
        # 加权融合，突出绿色植被特征
        enhanced = x * (1 + veg_weight * green_enhanced)
        
        return enhanced

class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module) - 增强版"""
    def __init__(self, in_channels, reduction=16, kernel_size=7, use_vegetation_attention=True):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
        # 新增：植被指数注意力，专门用于小目标幼苗检测
        self.use_vegetation_attention = use_vegetation_attention
        if use_vegetation_attention:
            self.vegetation_attention = VegetationIndexAttention(in_channels)
        
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        
        # 空间注意力
        x = x * self.spatial_attention(x)
        
        # 植被指数注意力 - 专门增强绿色植被特征
        if self.use_vegetation_attention:
            x = self.vegetation_attention(x)
        
        return x

class DetailEnhancementBlock(nn.Module):
    """细节增强块 - 专门提取细微差异特征"""
    def __init__(self, in_channels, out_channels):
        super(DetailEnhancementBlock, self).__init__()
        
        # 边缘检测分支
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 纹理检测分支 - 修复分组卷积问题
        # 确保groups能整除out_channels//2
        groups = min(max(1, in_channels//4), out_channels//2)
        # 进一步确保out_channels//2能被groups整除
        while (out_channels//2) % groups != 0 and groups > 1:
            groups -= 1
        
        self.texture_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=5, padding=2, groups=groups),
            nn.Conv2d(out_channels//2, out_channels//2, kernel_size=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力机制
        self.cbam = CBAM(out_channels)
        
    def forward(self, x):
        edge_feat = self.edge_conv(x)
        texture_feat = self.texture_conv(x)
        
        # 拼接特征
        combined = torch.cat([edge_feat, texture_feat], dim=1)
        fused = self.fusion(combined)
        
        # 应用注意力
        enhanced = self.cbam(fused)
        
        return enhanced

class ContrastiveLearningHead(nn.Module):
    """对比学习头 - 增强困难样本的特征区分度"""
    def __init__(self, feature_dim=512, projection_dim=128):
        super(ContrastiveLearningHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )
        
    def forward(self, features):
        return F.normalize(self.projection(features), dim=1)

class GrowthStageFeatureExtractor(nn.Module):
    """增强版玉米生育期特征提取器"""
    def __init__(self, base_model_name='efficientnet_b3', pretrained=True, use_attention=True, 
                 use_multiscale=True, use_difficult_enhancement=True):
        super(GrowthStageFeatureExtractor, self).__init__()
        
        self.use_attention = use_attention
        self.use_multiscale = use_multiscale
        self.use_difficult_enhancement = use_difficult_enhancement
        self.using_features_only = False  # 默认不使用features_only模式
        
        # 多尺度特征提取
        if use_multiscale:
            self.multiscale_extractor = MultiScaleFeatureExtractor(3)
        
        # 加载基础模型
        if 'efficientnet' in base_model_name:
            try:
                # 首先尝试使用timm库加载模型
                self.base_model = timm.create_model(base_model_name, pretrained=pretrained, features_only=True)
                feature_dims = self.base_model.feature_info.channels()
                self.feature_dim = feature_dims[-1]
                self.using_features_only = True
                print(f"成功使用timm库加载{base_model_name}模型")
            except Exception as e:
                print(f"从timm加载{base_model_name}失败: {str(e)}")
                print(f"切换到PyTorch内置的EfficientNet模型...")
                
                # 使用PyTorch内置的EfficientNet模型作为后备选项
                if base_model_name == 'efficientnet_b0':
                    from torchvision.models import EfficientNet_B0_Weights
                    self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b1':
                    from torchvision.models import EfficientNet_B1_Weights
                    self.base_model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b2':
                    from torchvision.models import EfficientNet_B2_Weights
                    self.base_model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b3' or base_model_name == 'efficientnet-b3':
                    from torchvision.models import EfficientNet_B3_Weights
                    self.base_model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b4':
                    from torchvision.models import EfficientNet_B4_Weights
                    self.base_model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b5':
                    from torchvision.models import EfficientNet_B5_Weights
                    self.base_model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b6':
                    from torchvision.models import EfficientNet_B6_Weights
                    self.base_model = models.efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None)
                elif base_model_name == 'efficientnet_b7':
                    from torchvision.models import EfficientNet_B7_Weights
                    self.base_model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None)
                else:
                    raise ValueError(f"不支持的EfficientNet类型: {base_model_name}")
                
                # 获取特征维度并删除分类器
                self.feature_dim = self.base_model.classifier[1].in_features
                self.base_model.classifier = nn.Identity()
                self.base_model.avgpool = nn.Identity()  # 不使用模型的平均池化
                print(f"成功切换到PyTorch内置的{base_model_name}模型")
        elif 'resnet' in base_model_name:
            if base_model_name == 'resnet18':
                from torchvision.models import ResNet18_Weights
                self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
                self.feature_dim = 512
            elif base_model_name == 'resnet34':
                from torchvision.models import ResNet34_Weights
                self.base_model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
                self.feature_dim = 512
            elif base_model_name == 'resnet50':
                from torchvision.models import ResNet50_Weights
                self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
                self.feature_dim = 2048
            elif base_model_name == 'resnet101':
                from torchvision.models import ResNet101_Weights
                self.base_model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
                self.feature_dim = 2048
            
            # 移除最后的全连接层和全局池化层
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
            print(f"成功加载{base_model_name}模型，特征维度: {self.feature_dim}")
        else:
            try:
                # 支持更多模型
                self.base_model = timm.create_model(base_model_name, pretrained=pretrained, features_only=False, num_classes=0)
                self.feature_dim = self.base_model.num_features
                print(f"成功使用timm库加载{base_model_name}模型，特征维度: {self.feature_dim}")
            except Exception as e:
                print(f"加载模型 {base_model_name} 失败: {str(e)}")
                print("回退到 resnet50...")
                self.base_model = models.resnet50(pretrained=pretrained)
                self.feature_dim = 2048
                self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
                print(f"成功回退到ResNet50模型，特征维度: {self.feature_dim}")
        
        # 多尺度分支与主干特征融合所需的对齐与融合层（仅在启用多尺度时创建）
        if self.use_multiscale:
            # 将多尺度融合输出（64通道）对齐到主干通道数
            self.ms_align = nn.Conv2d(64, self.feature_dim, kernel_size=1, bias=False)
            self.ms_align_bn = nn.BatchNorm2d(self.feature_dim)
            self.ms_align_act = nn.SiLU(inplace=True)

            # 拼接后通道回归到主干维度
            self.fuse_conv = nn.Conv2d(2 * self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
            self.fuse_bn = nn.BatchNorm2d(self.feature_dim)
            self.fuse_act = nn.SiLU(inplace=True)

        # 细节增强模块
        if use_difficult_enhancement:
            self.detail_enhancer = DetailEnhancementBlock(self.feature_dim, 512)
            
            # 困难样本特征增强
            self.difficult_stage_enhancer = nn.ModuleDict({
                'early_stage': self._create_stage_enhancer(512, 256),  # 11, 21
                'late_stage': self._create_stage_enhancer(512, 256),   # 81, 91
                'general': self._create_stage_enhancer(512, 256)       # 其他阶段
            })
            
            # 困难样本判别器
            self.difficulty_classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 3)  # early_stage, late_stage, general
            )
            
            # 特征融合
            self.feature_fusion = nn.Sequential(
                nn.Conv2d(512 + 256, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            
            self.enhanced_feature_dim = 512
        else:
            self.enhanced_feature_dim = self.feature_dim
        
        # 注意力模块 (CBAM)
        if use_attention:
            self.channel_attention = ChannelAttention(self.enhanced_feature_dim)
            self.spatial_attention = SpatialAttention()
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _create_stage_enhancer(self, in_channels, out_channels):
        """创建阶段特定的特征增强器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # 统一先提取主干特征
        if self.using_features_only:
            base_feat = self.base_model(x)[-1]
        else:
            base_feat = self.base_model(x)

        # 多尺度分支与主干融合
        if self.use_multiscale:
            ms = self.multiscale_extractor(x)  # [B, 64, H, W]
            # 对齐空间尺寸，尽量使用抗混叠（旧版PyTorch不支持时回退）
            try:
                ms = F.interpolate(ms, size=base_feat.shape[-2:], mode='bilinear', align_corners=False, antialias=True)
            except TypeError:
                ms = F.interpolate(ms, size=base_feat.shape[-2:], mode='bilinear', align_corners=False)

            ms = self.ms_align_act(self.ms_align_bn(self.ms_align(ms)))  # [B, Cb, h, w]
            fused = torch.cat([base_feat, ms], dim=1)                    # [B, 2Cb, h, w]
            features = self.fuse_act(self.fuse_bn(self.fuse_conv(fused))) # [B, Cb, h, w]
        else:
            features = base_feat
        
        # 困难样本增强
        if self.use_difficult_enhancement:
            # 细节增强
            enhanced_feat = self.detail_enhancer(features)
            
            # 全局平均池化用于困难样本判别
            global_feat = F.adaptive_avg_pool2d(enhanced_feat, (1, 1)).view(batch_size, -1)
            
            # 困难样本分类
            difficulty_logits = self.difficulty_classifier(global_feat)
            difficulty_probs = F.softmax(difficulty_logits, dim=1)
            
            # 根据困难程度选择特征增强器
            early_enhanced = self.difficult_stage_enhancer['early_stage'](enhanced_feat)
            late_enhanced = self.difficult_stage_enhancer['late_stage'](enhanced_feat)
            general_enhanced = self.difficult_stage_enhancer['general'](enhanced_feat)
            
            # 加权融合不同增强器的输出
            early_weight = difficulty_probs[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            late_weight = difficulty_probs[:, 1:2].unsqueeze(-1).unsqueeze(-1)
            general_weight = difficulty_probs[:, 2:3].unsqueeze(-1).unsqueeze(-1)
            
            adaptive_enhanced = (early_weight * early_enhanced + 
                               late_weight * late_enhanced + 
                               general_weight * general_enhanced)
            
            # 特征融合
            combined_feat = torch.cat([enhanced_feat, adaptive_enhanced], dim=1)
            features = self.feature_fusion(combined_feat)
        
        # 应用注意力 (CBAM)
        if self.use_attention:
            # 通道注意力
            features = features * self.channel_attention(features)
            
            # 空间注意力
            features = features * self.spatial_attention(features)
        
        # 全局池化
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        if return_features:
            feature_dict = {
                'final_features': flattened,
                'spatial_features': features
            }
            if self.use_difficult_enhancement:
                feature_dict['difficulty_probs'] = difficulty_probs
            return flattened, feature_dict
        
        return flattened

class ImprovedCornGrowthStageModel(nn.Module):
    """改进版玉米生长阶段识别模型"""
    
    def __init__(self, 
                 num_classes=9, 
                 model_type='efficientnet_b3', 
                 pretrained=True, 
                 dropout_rate=0.5, 
                 include_time=True,
                 use_attention=True,
                 use_contrastive=True,
                 use_multiscale=None,      # 新增参数
                 use_difficult_enhancement=None):  # 新增参数
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            model_type: 模型类型
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比率
            include_time: 是否包含时间特征
            use_attention: 是否使用注意力机制
            use_contrastive: 是否使用对比学习
            use_multiscale: 是否使用多尺度特征提取 (None时使用CONFIG)
            use_difficult_enhancement: 是否使用困难样本增强 (None时使用CONFIG)
        """
        super(ImprovedCornGrowthStageModel, self).__init__()
        
        self.model_type = model_type
        self.include_time = include_time
        self.use_attention = use_attention
        self.use_contrastive = use_contrastive
        
        # 参数优先级：传入参数 > CONFIG默认值
        self.use_multiscale = use_multiscale if use_multiscale is not None else CONFIG['use_multiscale_features']
        self.use_difficult_enhancement = use_difficult_enhancement if use_difficult_enhancement is not None else CONFIG['use_difficult_stage_enhancement']
        
        print(f"🔧 模型组件配置:")
        print(f"  ✅ 时间特征: {self.include_time}")
        print(f"  ✅ 注意力机制: {self.use_attention}")
        print(f"  ✅ 多尺度特征: {self.use_multiscale}")
        print(f"  ✅ 困难样本增强: {self.use_difficult_enhancement}")
        print(f"  ✅ 对比学习: {self.use_contrastive}")
        
        # 特征提取器 - 使用统一的参数
        self.feature_extractor = GrowthStageFeatureExtractor(
            base_model_name=model_type,
            pretrained=pretrained,
            use_attention=self.use_attention,
            use_multiscale=self.use_multiscale,
            use_difficult_enhancement=self.use_difficult_enhancement
        )
        
        self.feature_dim = self.feature_extractor.enhanced_feature_dim if self.use_difficult_enhancement else self.feature_extractor.feature_dim
        print(f"  📏 特征维度: {self.feature_dim}")
        
        # 时间特征处理
        if include_time:
            self.time_module = TimeFeatureModule(
                time_dim=3,  # 只用3维时间特征
                hidden_dim=128,
                dropout_rate=dropout_rate
            )
            
            # 特征融合
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim + 128, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            final_dim = 512
            print(f"  🔗 融合后维度: {final_dim} (图像: {self.feature_dim} + 时间: 128)")
        else:
            final_dim = self.feature_dim
            print(f"  🖼️ 最终维度: {final_dim} (仅图像特征)")
        
        # 对比学习头
        if use_contrastive:
            self.contrastive_head = ContrastiveLearningHead(final_dim, 128)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
        # 随机深度模块 - 也使用参数控制
        use_stochastic_depth = CONFIG.get('use_stochastic_depth', False)
        if use_stochastic_depth:
            self.stochastic_depth = StochasticDepth(CONFIG.get('stochastic_depth_prob', 0.2))
        else:
            self.stochastic_depth = None
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_contrastive=False):
        if self.include_time:
            # 分离图像和时间特征
            if isinstance(x, list):
                img, time_features = x[0], x[1]
            elif isinstance(x, tuple):
                img, time_features = x
            else:
                raise ValueError("时间特征模式下，输入必须是元组或列表 (img, time_features)")
            
            # 提取图像特征
            img_features, feature_dict = self.feature_extractor(img, return_features=True)
            
            # 处理时间特征
            time_features = self.time_module(time_features)
            
            # 融合特征
            combined_features = torch.cat([img_features, time_features], dim=1)
            fused_features = self.fusion(combined_features)
            
            # 分类
            logits = self.classifier(fused_features)
            
            # 对比学习特征
            if return_contrastive and self.use_contrastive:
                contrastive_features = self.contrastive_head(fused_features)
                return logits, contrastive_features, feature_dict
            elif return_contrastive:
                return logits, None, feature_dict
            
        else:
            # 只处理图像
            features, feature_dict = self.feature_extractor(x, return_features=True)
            logits = self.classifier(features)
            
            # 对比学习特征
            if return_contrastive and self.use_contrastive:
                contrastive_features = self.contrastive_head(features)
                return logits, contrastive_features, feature_dict
            elif return_contrastive:
                return logits, None, feature_dict
        
        return logits

# 困难样本对比学习损失函数
class DifficultStageContrastiveLoss(nn.Module):
    """困难生育期对比学习损失"""
    def __init__(self, temperature=0.07, difficult_stages=[1, 7], weight_factor=2.0):
        super(DifficultStageContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.difficult_stages = set(difficult_stages)  # 21, 91 对应的索引 1, 7
        self.weight_factor = weight_factor
        
    def forward(self, features, labels):
        """
        Args:v
            features: 对比学习特征 [batch_size, feature_dim]
            labels: 真实标签 [batch_size]
        """
        if features is None:
            return torch.tensor(0.0, device=labels.device)
            
        batch_size = features.size(0)
        device = features.device
        
        # 计算相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 识别困难样本
        difficult_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for i, label in enumerate(labels.squeeze()):
            if label.item() in self.difficult_stages:
                difficult_mask[i] = True
        
        # 计算损失
        # 移除对角线元素
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 数值稳定性
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # 计算对数概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # 计算平均对数似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # 对困难样本加权
        weights = torch.ones(batch_size, device=device)
        weights[difficult_mask] = self.weight_factor
        
        # 损失
        loss = -(weights * mean_log_prob_pos).mean()
        
        return loss

# 组合损失函数
class CombinedLossFunction(nn.Module):
    """组合损失函数：分类损失 + 对比学习损失 + 小目标损失"""
    def __init__(self, num_classes=9, use_focal=True, gamma=2.0, 
                 contrastive_weight=0.1, small_target_weight=0.3, class_weights=None):
        super(CombinedLossFunction, self).__init__()
        
        self.contrastive_weight = contrastive_weight
        self.small_target_weight = small_target_weight
        
        # 分类损失
        if use_focal:
            self.classification_loss = FocalLoss(gamma=gamma, alpha=class_weights)
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # 对比学习损失
        self.contrastive_loss = DifficultStageContrastiveLoss()
        
        # 小目标专用损失 - 专门针对播种期(0)和出苗期(1)
        self.small_target_loss = SmallTargetLoss(
            num_classes=num_classes, 
            small_target_classes=[0, 1],  # 播种期和出苗期的索引
            alpha=2.0  # 2倍权重
        )
    
    def forward(self, logits, labels, contrastive_features=None):
        # 分类损失
        cls_loss = self.classification_loss(logits, labels)
        
        # 小目标损失
        small_target_loss = self.small_target_loss(logits, labels)
        
        # 对比学习损失
        if contrastive_features is not None:
            con_loss = self.contrastive_loss(contrastive_features, labels)
            total_loss = (cls_loss + 
                         self.small_target_weight * small_target_loss + 
                         self.contrastive_weight * con_loss)
            return total_loss, {
                'cls_loss': cls_loss.item(), 
                'con_loss': con_loss.item(),
                'small_target_loss': small_target_loss.item()
            }
        else:
            total_loss = cls_loss + self.small_target_weight * small_target_loss
            return total_loss, {
                'cls_loss': cls_loss.item(), 
                'con_loss': 0.0,
                'small_target_loss': small_target_loss.item()
            }

class SmallTargetLoss(nn.Module):
    """小目标专用损失函数 - 专门针对播种期和出苗期"""
    def __init__(self, num_classes=9, small_target_classes=[0, 1], alpha=2.0):
        super(SmallTargetLoss, self).__init__()
        self.num_classes = num_classes
        self.small_target_classes = set(small_target_classes)  # 播种期(0)和出苗期(1)
        self.alpha = alpha
        
    def forward(self, logits, targets):
        # 基础交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # 对小目标类别加权
        weights = torch.ones_like(ce_loss)
        for i, target in enumerate(targets):
            if target.item() in self.small_target_classes:
                weights[i] = self.alpha
        
        weighted_ce_loss = (weights * ce_loss).mean()
        
        return weighted_ce_loss

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡问题"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失函数"""
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)

        if self.weight is not None:
            log_prob = log_prob * self.weight.unsqueeze(0)

        nll_loss = -log_prob.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        smooth_loss = -log_prob.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        return loss.mean()

class StochasticDepth(nn.Module):
    """随机深度模块 - 训练时随机跳过某些层"""
    def __init__(self, prob=0.2):
        super(StochasticDepth, self).__init__()
        self.prob = prob
    
    def forward(self, x, residual=None):
        if not self.training:
            return x
        
        if torch.rand(1).item() < self.prob:
            # 随机跳过当前层
            return residual if residual is not None else x
        else:
            return x

class MixupCutmixCallback:
    """Mixup和Cutmix数据增强回调"""
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
    
    def __call__(self, batch, targets):
        # 决定是否应用mixup/cutmix
        if np.random.rand() >= self.prob:
            # 当不应用mixup/cutmix时，仍然返回合适的三元组格式
            # 使用lam=1.0，这样全部权重给原始标签
            return batch, (targets, targets, 1.0)
        
        # 决定使用mixup还是cutmix
        use_cutmix = np.random.rand() >= self.switch_prob
        
        batch_size = batch.size(0)
        indices = torch.randperm(batch_size).to(batch.device)
        
        if use_cutmix:
            # Cutmix
            alpha = self.cutmix_alpha
            lam = np.random.beta(alpha, alpha)
            
            # 生成随机裁剪框
            y1, y2, x1, x2 = self._rand_bbox(batch.size(), lam)
            batch[:, :, y1:y2, x1:x2] = batch[indices, :, y1:y2, x1:x2]
            
            # 调整混合比例（面积占比，W 对应 dim=-1，H 对应 dim=-2）
            lam = 1 - ((x2 - x1) * (y2 - y1) / (batch.size(-1) * batch.size(-2)))
            
        else:
            # Mixup
            alpha = self.mixup_alpha
            lam = np.random.beta(alpha, alpha)
            batch = lam * batch + (1 - lam) * batch[indices]
        
        # 转换lam为标量值，避免类型不匹配
        lam_value = float(lam)
        
        # 明确构造一个元组作为返回值
        mixed_targets = (targets, targets[indices], lam_value)
        
        # 返回混合后的batch和目标
        return batch, mixed_targets
    
    def _rand_bbox(self, size, lam):
        """生成随机裁剪框（遵循 NCHW: size = [N, C, H, W]）"""
        _, _, H, W = size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # 随机中心点（x 对应宽度维，y 对应高度维）
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 限制范围，返回顺序与切片维度匹配：y1:y2 对应 dim=2, x1:x2 对应 dim=3
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        return y1, y2, x1, x2

class CornDatasetBalanced(Dataset):
    """平衡的玉米数据集，支持困难样本增强和平衡采样"""
    
    def __init__(self, dataframe, transform=None, include_time=True, 
                 balanced_sampling=False, class_weights_power=0.5,
                 augment_rare_classes=True, rare_class_threshold=0.1,
                 img_size=224):  # 添加img_size参数
        self.df = dataframe.copy()
        self.transform = transform
        self.include_time = include_time
        self.balanced_sampling = balanced_sampling
        self.augment_rare_classes = augment_rare_classes
        self.rare_class_threshold = rare_class_threshold
        self.img_size = img_size  # 保存图像尺寸
        
        # 计算类别权重用于平衡采样
        if balanced_sampling:
            self.class_weights = self._calculate_sampling_weights(class_weights_power)
            self.sample_weights = self._get_sample_weights()
        
        # 识别罕见类别
        if augment_rare_classes:
            self.rare_classes = self._identify_rare_classes()
        else:
            self.rare_classes = set()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取样本索引
        if self.balanced_sampling:
            # 使用加权采样
            actual_idx = torch.multinomial(self.sample_weights, 1).item()
        else:
            actual_idx = idx
        
        row = self.df.iloc[actual_idx]
        img_path = row['path']
        
        # 加载图像
        try:
            img = self._load_image(img_path)
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {str(e)}")
            # 创建一个默认的黑色图像
            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        
        # 对罕见类别应用更强的数据增强
        if self.augment_rare_classes and row['stage_idx'] in self.rare_classes:
            try:
                img = self._apply_stronger_augmentation(img)
            except Exception as e:
                print(f"强化增强失败: {img_path}, 错误: {str(e)}")
                # 回退到基础变换
                if self.transform:
                    img = self.transform(img)
                else:
                    basic_transform = transforms.Compose([
                        transforms.Resize((self.img_size, self.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img = basic_transform(img)
        else:
            # 应用标准变换
            if self.transform:
                img = self.transform(img)
            else:
                basic_transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img = basic_transform(img)
        
        # 确保图像是正确的tensor格式
        if not isinstance(img, torch.Tensor):
            print(f"警告: 图像不是tensor格式，路径: {img_path}")
            # 强制转换为tensor
            if isinstance(img, Image.Image):
                img = transforms.ToTensor()(img)
            else:
                # 创建一个默认的tensor
                img = torch.zeros(3, self.img_size, self.img_size)
        
        # 确保图像尺寸正确
        expected_shape = (3, self.img_size, self.img_size)
        if img.shape != expected_shape:
            print(f"警告: 图像尺寸不正确 {img.shape}，期望 {expected_shape}，路径: {img_path}")
            # 强制resize
            img = F.interpolate(img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False).squeeze(0)
        
        # 获取标签
        label = row['stage_idx']
        
        # 处理时间特征
        if self.include_time:
            # 使用连续的年内时间表示
            time_in_year = ((row['month'] - 1) * 31 + row['day']) / (12 * 31)  # 年内时间进度 [0,1]
            import math
            time_angle = 2 * math.pi * time_in_year  # 将时间转换为角度
            time_sin = math.sin(time_angle)  # 正弦分量
            time_cos = math.cos(time_angle)  # 余弦分量
            # 只保留3维时间特征
            time_feature = torch.tensor([
                time_in_year, time_sin, time_cos
            ], dtype=torch.float32)
            return (img, time_feature), label
        else:
            return img, label
    
    def _load_image(self, path):
        """加载图像"""
        img = Image.open(path).convert('RGB')
        return img
    
    def _apply_stronger_augmentation(self, img):
        """对罕见类别应用更强的数据增强"""
        # 创建更强的数据增强 - 使用动态图像尺寸
        stronger_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            # 移除RandomGrayscale以避免PIL错误
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return stronger_transform(img)
    
    def _get_season_encoding(self, month):
        """将月份转换为季节的one-hot编码"""
        # 春季(3-5月), 夏季(6-8月), 秋季(9-11月), 冬季(12-2月)
        if 3 <= month <= 5:
            return [1.0, 0.0, 0.0, 0.0]  # 春季
        elif 6 <= month <= 8:
            return [0.0, 1.0, 0.0, 0.0]  # 夏季
        elif 9 <= month <= 11:
            return [0.0, 0.0, 1.0, 0.0]  # 秋季
        else:  # 12, 1, 2月
            return [0.0, 0.0, 0.0, 1.0]  # 冬季
    
    def _calculate_sampling_weights(self, class_weights_power):
        """计算类别权重用于平衡采样"""
        class_counts = self.df['stage_idx'].value_counts()
        # 按照 1/freq^power 计算权重
        weights = (1.0 / class_counts) ** class_weights_power
        # 归一化权重
        weights = weights / weights.sum()
        return weights
    
    def _get_sample_weights(self):
        """为每个样本分配权重"""
        sample_weights = self.df['stage_idx'].map(lambda x: self.class_weights[x]).values
        # 确保权重总和为1
        sample_weights = sample_weights / sample_weights.sum()
        return torch.tensor(sample_weights, dtype=torch.float32)
    
    def _identify_rare_classes(self):
        """识别罕见类别"""
        class_counts = self.df['stage_idx'].value_counts()
        total_samples = len(self.df)
        rare_classes = set(class_counts[class_counts / total_samples < self.rare_class_threshold].index)
        return rare_classes

class ClampTransform:
    """可序列化的数值范围限制变换"""
    def __init__(self, min_val=-3.0, max_val=3.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val)

def get_transforms(config, is_training=True, small_target_mode=False):
    """获取图像转换 - 数值稳定版本（可序列化）
    
    Args:
        config: 配置字典
        is_training: 是否为训练模式
        small_target_mode: 是否启用小目标保护模式
    """
    img_size = config['img_size']
    
    if is_training:
        # 简化的训练时数据增强 - 避免极端变换
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 先resize到目标尺寸
            transforms.RandomHorizontalFlip(p=0.5),  # 简单的水平翻转
            
            # 保守的颜色变换 - 避免数值异常
            transforms.ColorJitter(
                brightness=0.1,  # 减小亮度变化范围
                contrast=0.1,    # 减小对比度变化范围
                saturation=0.1,  # 减小饱和度变化范围
                hue=0.05         # 减小色调变化范围
            ),
            
            # 轻微的几何变换
            transforms.RandomAffine(
                degrees=5,            # 小角度旋转
                translate=(0.02, 0.02),  # 微小平移
                scale=(0.98, 1.02),   # 微小缩放
                shear=2               # 微小剪切
            ),
            
            transforms.ToTensor(),
            
            # 标准化 - 使用ImageNet预训练模型的标准值
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            
            # 使用可序列化的类替代lambda函数
            ClampTransform(min_val=-3.0, max_val=3.0)
        ])
    else:
        # 验证时的简单变换
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            # 同样使用可序列化的类
            ClampTransform(min_val=-3.0, max_val=3.0)
        ])

def calculate_class_weights(y_train, power=0.5):
    """计算类别权重来处理不平衡数据"""
    class_counts = np.bincount(y_train)
    total = len(y_train)
    
    # 按照 1/freq^power 计算权重
    weights = (total / (len(class_counts) * class_counts)) ** power
    
    # 标准化权重
    weights = weights / np.sum(weights) * len(class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, dataloader, criterion, optimizer, device, 
                    scheduler=None, use_amp=True, use_mixup=False, 
                    mixup_cutmix=None, use_contrastive=True, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_con_loss = 0.0
    correct = 0
    total = 0
    
    # 困难样本统计
    difficult_correct = {'21': 0, '91': 0}
    difficult_total = {'21': 0, '91': 0}
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    
    # 使用tqdm创建进度条
    try:
        from tqdm import tqdm
        dataloader = tqdm(dataloader, desc="训练中", ncols=100)
    except ImportError:
        print("警告: 未安装tqdm，将不显示进度条")
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        try:
            # 检查输入数据的有效性
            if isinstance(inputs, tuple):
                img, time_features = inputs
                if torch.isnan(img).any() or torch.isinf(img).any():
                    print(f"警告: batch {batch_idx} 图像数据包含NaN或Inf，跳过")
                    continue
                if torch.isnan(time_features).any() or torch.isinf(time_features).any():
                    print(f"警告: batch {batch_idx} 时间特征包含NaN或Inf，跳过")
                    continue
            elif isinstance(inputs, torch.Tensor):
                if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                    print(f"警告: batch {batch_idx} 输入数据包含NaN或Inf，跳过")
                    continue
            
            # 检查标签的有效性
            if torch.isnan(labels.float()).any() or (labels < 0).any() or (labels >= 9).any():
                print(f"警告: batch {batch_idx} 标签无效，跳过")
                continue
            
            # 移动数据到设备
            if isinstance(inputs, list):
                # 处理列表类型的输入
                inputs = [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in inputs]
            elif isinstance(inputs, tuple):
                inputs = (inputs[0].to(device, non_blocking=True), inputs[1].to(device, non_blocking=True))
            else:
                inputs = inputs.to(device, non_blocking=True)
                
            labels = labels.to(device, non_blocking=True)
            
            # 应用Mixup/Cutmix
            mixed_labels = None
            if use_mixup and mixup_cutmix is not None:
                try:
                    if isinstance(inputs, tuple):
                        img, time_features = inputs
                        img, mixed_result = mixup_cutmix(img, labels)
                        inputs = (img, time_features)
                    elif isinstance(inputs, list):
                        # 对列表中的第一个元素（通常是图像）应用mixup
                        inputs[0], mixed_result = mixup_cutmix(inputs[0], labels)
                    else:
                        inputs, mixed_result = mixup_cutmix(inputs, labels)
                    
                    # 处理不同类型的返回值
                    if isinstance(mixed_result, tuple) and len(mixed_result) == 3:
                        # 正确的元组形式
                        mixed_labels = mixed_result
                    else:
                        mixed_labels = None
                        
                except Exception as e:
                    print(f"batch {batch_idx}: mixup/cutmix应用失败: {str(e)}")
                    mixed_labels = None
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播（混合精度）
            if use_amp:
                with autocast():
                    try:
                        if use_contrastive:
                            outputs, contrastive_features, feature_dict = model(inputs, return_contrastive=True)
                        else:
                            outputs = model(inputs)
                            contrastive_features = None
                        
                        # 检查输出的有效性
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"警告: batch {batch_idx} 模型输出包含NaN或Inf，跳过")
                            continue
                        
                        # 计算损失
                        if mixed_labels is not None:
                            # Mixup/Cutmix损失
                            targets1, targets2, lam = mixed_labels
                            if isinstance(criterion, CombinedLossFunction):
                                loss1, loss_dict1 = criterion(outputs, targets1, contrastive_features)
                                loss2, loss_dict2 = criterion(outputs, targets2, contrastive_features)
                                loss = lam * loss1 + (1 - lam) * loss2
                                loss_dict = {
                                    'cls_loss': lam * loss_dict1['cls_loss'] + (1 - lam) * loss_dict2['cls_loss'],
                                    'con_loss': lam * loss_dict1['con_loss'] + (1 - lam) * loss_dict2['con_loss']
                                }
                            else:
                                loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
                                loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                        else:
                            # 普通损失
                            if isinstance(criterion, CombinedLossFunction):
                                loss, loss_dict = criterion(outputs, labels, contrastive_features)
                            else:
                                loss = criterion(outputs, labels)
                                loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                        
                        # 检查损失的有效性
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"警告: batch {batch_idx} 损失为NaN或Inf，跳过")
                            continue
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e) or "CUDA" in str(e):
                            print(f"CUDA错误在前向传播: {str(e)}")
                            # 清理GPU缓存
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                
                # 反向传播
                try:
                    scaler.scale(loss).backward()
                    
                    # 检查梯度的有效性
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"警告: 参数 {name} 的梯度包含NaN或Inf")
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        print(f"batch {batch_idx}: 检测到NaN梯度，跳过此批次")
                        optimizer.zero_grad()
                        continue
                    
                    # 添加梯度裁剪，防止梯度爆炸
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    
                    # 检查梯度范数
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"警告: batch {batch_idx} 梯度范数为NaN或Inf，跳过")
                        optimizer.zero_grad()
                        continue
                    
                    # 优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        print(f"CUDA错误在反向传播: {str(e)}")
                        # 清理GPU缓存
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        print(f"反向传播错误: {str(e)}")
                        optimizer.zero_grad()
                        continue
            else:
                # 不使用混合精度的类似处理
                try:
                    if use_contrastive:
                        outputs, contrastive_features, feature_dict = model(inputs, return_contrastive=True)
                    else:
                        outputs = model(inputs)
                        contrastive_features = None
                    
                    # 检查输出的有效性
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"警告: batch {batch_idx} 模型输出包含NaN或Inf，跳过")
                        continue
                    
                    # 计算损失
                    if mixed_labels is not None:
                        targets1, targets2, lam = mixed_labels
                        if isinstance(criterion, CombinedLossFunction):
                            loss1, loss_dict1 = criterion(outputs, targets1, contrastive_features)
                            loss2, loss_dict2 = criterion(outputs, targets2, contrastive_features)
                            loss = lam * loss1 + (1 - lam) * loss2
                            loss_dict = {
                                'cls_loss': lam * loss_dict1['cls_loss'] + (1 - lam) * loss_dict2['cls_loss'],
                                'con_loss': lam * loss_dict1['con_loss'] + (1 - lam) * loss_dict2['con_loss']
                            }
                        else:
                            loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
                            loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                    else:
                        if isinstance(criterion, CombinedLossFunction):
                            loss, loss_dict = criterion(outputs, labels, contrastive_features)
                        else:
                            loss = criterion(outputs, labels)
                            loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                    
                    # 检查损失的有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: batch {batch_idx} 损失为NaN或Inf，跳过")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    
                    # 检查梯度的有效性
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"警告: 参数 {name} 的梯度包含NaN或Inf")
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        print(f"batch {batch_idx}: 检测到NaN梯度，跳过此批次")
                        optimizer.zero_grad()
                        continue
                    
                    # 添加梯度裁剪，防止梯度爆炸
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    
                    # 检查梯度范数
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"警告: batch {batch_idx} 梯度范数为NaN或Inf，跳过")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        print(f"CUDA错误: {str(e)}")
                        # 清理GPU缓存
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        print(f"训练错误: {str(e)}")
                        optimizer.zero_grad()
                        continue
            
            # 更新学习率
            if scheduler is not None and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # 统计
            running_loss += loss.item()
            running_cls_loss += loss_dict['cls_loss']
            running_con_loss += loss_dict['con_loss']
            
            _, predicted = torch.max(outputs, 1)
            
            # 在使用mixup时，只对原始标签计算准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 困难样本统计
            for i, (pred, target) in enumerate(zip(predicted, labels)):
                if target.item() == 1:  # 出苗期(21)
                    difficult_total['21'] += 1
                    if pred == target:
                        difficult_correct['21'] += 1
                elif target.item() == 7:  # 成熟期(91)
                    difficult_total['91'] += 1
                    if pred == target:
                        difficult_correct['91'] += 1
            
            # 更新进度条
            if hasattr(dataloader, "set_postfix"):
                # 如果dataloader是tqdm封装的，更新信息
                current_loss = running_loss / (batch_idx + 1)
                current_acc = correct / total if total > 0 else 0.0
                current_cls_loss = running_cls_loss / (batch_idx + 1)
                current_con_loss = running_con_loss / (batch_idx + 1)
                
                postfix_dict = {
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.4f}',
                    'cls': f'{current_cls_loss:.4f}'
                }
                if use_contrastive:
                    postfix_dict['con'] = f'{current_con_loss:.4f}'
                
                dataloader.set_postfix(postfix_dict)
                
        except Exception as e:
            print(f"batch {batch_idx} 处理失败: {str(e)}")
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue
    
    # 更新学习率（非OneCycleLR）
    if scheduler is not None and not isinstance(scheduler, OneCycleLR):
        scheduler.step()
    

    # 防止除零错误
    if len(dataloader) == 0:
        return 0.0, 0.0, 0.0, 0.0, {}
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    epoch_cls_loss = running_cls_loss / len(dataloader)
    epoch_con_loss = running_con_loss / len(dataloader)
    
    # 困难样本准确率
    difficult_accs = {}
    for stage in ['21', '91']:
        if difficult_total[stage] > 0:
            acc = difficult_correct[stage] / difficult_total[stage]
            difficult_accs[stage] = acc
    
    return epoch_loss, epoch_acc, epoch_cls_loss, epoch_con_loss, difficult_accs

def validate(model, dataloader, criterion, device, use_amp=True, use_contrastive=True):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_con_loss = 0.0
    correct = 0
    total = 0
    
    # 困难样本统计
    difficult_correct = {'21': 0, '91': 0}
    difficult_total = {'21': 0, '91': 0}
    
    all_labels = []
    all_preds = []
    
    # 保存原始dataloader以访问dataset属性
    original_dataloader = dataloader
    
    # 使用tqdm创建进度条
    try:
        from tqdm import tqdm
        dataloader = tqdm(dataloader, desc="验证中", ncols=100)
    except ImportError:
        pass
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            try:
                # 检查输入数据的有效性
                if isinstance(inputs, tuple):
                    img, time_features = inputs
                    if torch.isnan(img).any() or torch.isinf(img).any():
                        print(f"警告: 验证batch {batch_idx} 图像数据包含NaN或Inf，跳过")
                        continue
                    if torch.isnan(time_features).any() or torch.isinf(time_features).any():
                        print(f"警告: 验证batch {batch_idx} 时间特征包含NaN或Inf，跳过")
                        continue
                elif isinstance(inputs, torch.Tensor):
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        print(f"警告: 验证batch {batch_idx} 输入数据包含NaN或Inf，跳过")
                        continue
                
                # 检查标签的有效性
                if torch.isnan(labels.float()).any() or (labels < 0).any() or (labels >= 9).any():
                    print(f"警告: 验证batch {batch_idx} 标签无效，跳过")
                    continue
                
                # 移动数据到设备
                if isinstance(inputs, list):
                    inputs = [x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in inputs]
                elif isinstance(inputs, tuple):
                    inputs = (inputs[0].to(device, non_blocking=True), inputs[1].to(device, non_blocking=True))
                else:
                    inputs = inputs.to(device, non_blocking=True)
                    
                labels = labels.to(device, non_blocking=True)
                
                # 前向传播
                try:
                    if use_amp:
                        with autocast():
                            if use_contrastive:
                                outputs, contrastive_features, feature_dict = model(inputs, return_contrastive=True)
                            else:
                                outputs = model(inputs)
                                contrastive_features = None
                            
                            # 检查输出的有效性
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                print(f"警告: 验证batch {batch_idx} 模型输出包含NaN或Inf，跳过")
                                continue
                            
                            # 计算损失
                            if isinstance(criterion, CombinedLossFunction):
                                loss, loss_dict = criterion(outputs, labels, contrastive_features)
                            else:
                                loss = criterion(outputs, labels)
                                loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                    else:
                        if use_contrastive:
                            outputs, contrastive_features, feature_dict = model(inputs, return_contrastive=True)
                        else:
                            outputs = model(inputs)
                            contrastive_features = None
                        
                        # 检查输出的有效性
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            print(f"警告: 验证batch {batch_idx} 模型输出包含NaN或Inf，跳过")
                            continue
                        
                        # 计算损失
                        if isinstance(criterion, CombinedLossFunction):
                            loss, loss_dict = criterion(outputs, labels, contrastive_features)
                        else:
                            loss = criterion(outputs, labels)
                            loss_dict = {'cls_loss': loss.item(), 'con_loss': 0.0}
                    
                    # 检查损失的有效性
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: 验证batch {batch_idx} 损失为NaN或Inf，跳过")
                        continue
                        
                except RuntimeError as e:
                    if "out of memory" in str(e) or "CUDA" in str(e):
                        print(f"验证CUDA错误: {str(e)}")
                        # 清理GPU缓存
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"验证前向传播错误: {str(e)}")
                        continue
                
                # 统计
                running_loss += loss.item()
                running_cls_loss += loss_dict['cls_loss']
                running_con_loss += loss_dict['con_loss']
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 困难样本统计
                for i, (pred, target) in enumerate(zip(predicted, labels)):
                    if target.item() == 1:  # 出苗期(21)
                        difficult_total['21'] += 1
                        if pred == target:
                            difficult_correct['21'] += 1
                    elif target.item() == 7:  # 成熟期(91)
                        difficult_total['91'] += 1
                        if pred == target:
                            difficult_correct['91'] += 1
                
                # 收集预测结果用于详细分析
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                # 更新进度条
                if hasattr(dataloader, "set_postfix"):
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = correct / total if total > 0 else 0.0
                    current_cls_loss = running_cls_loss / (batch_idx + 1)
                    current_con_loss = running_con_loss / (batch_idx + 1)
                    
                    postfix_dict = {
                        'loss': f'{current_loss:.4f}',
                        'acc': f'{current_acc:.4f}',
                        'cls': f'{current_cls_loss:.4f}'
                    }
                    if use_contrastive:
                        postfix_dict['con'] = f'{current_con_loss:.4f}'
                    
                    dataloader.set_postfix(postfix_dict)
                    
            except Exception as e:
                print(f"验证batch {batch_idx} 处理失败: {str(e)}")
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
    
    # 防止除零错误
    if len(original_dataloader) == 0:
        return 0.0, 0.0, 0.0, 0.0, {}, [], []
    
    # 使用原始dataloader计算最终指标
    epoch_loss = running_loss / len(original_dataloader)
    epoch_acc = correct / total if total > 0 else 0.0
    epoch_cls_loss = running_cls_loss / len(original_dataloader)
    epoch_con_loss = running_con_loss / len(original_dataloader)
    
    # 困难样本准确率
    difficult_accs = {}
    for stage in ['21', '91']:
        if difficult_total[stage] > 0:
            acc = difficult_correct[stage] / difficult_total[stage]
            difficult_accs[stage] = acc
    
    return epoch_loss, epoch_acc, epoch_cls_loss, epoch_con_loss, difficult_accs, all_labels, all_preds

def get_lr_scheduler(optimizer, train_loader, config):
    """获取学习率调度器"""
    num_epochs = config['epochs']
    
    # 使用OneCycleLR可以获得更好的训练结果
    return OneCycleLR(
        optimizer,
        max_lr=config['base_lr'],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0
    )

def save_model(model, optimizer, epoch, config, stage_mapping, val_acc, val_loss, best_val_acc, output_dir):
    """保存模型"""
    try:
        print(f"尝试保存模型，周期: {epoch}, 准确率: {val_acc:.4f}")
        
        # 创建检查点字典
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'config': config,
            'stage_mapping': stage_mapping
        }
        
        # 尝试使用内存缓冲区保存
        # 规范化路径
        output_dir = str(pathlib.Path(output_dir))
        
        # 创建文件名
        model_filename = f"improved_corn_model_epoch{epoch:03d}_acc{val_acc:.4f}.pth"
        model_path = os.path.join(output_dir, model_filename)
        
        # 确保输出目录存在 - 尝试不同的方法创建目录
        try:
            print(f"尝试创建目录: {output_dir}")
            # 方法1: 使用os.makedirs
            os.makedirs(output_dir, exist_ok=True)
            
            # 方法2: 使用pathlib
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            print(f"目录创建成功? {os.path.exists(output_dir)}")
        except Exception as e:
            print(f"创建目录失败: {str(e)}")
            
            # 尝试使用不带中文字符的备用路径
            try:
                # 使用C盘或用户目录
                import tempfile
                output_dir = os.path.join(tempfile.gettempdir(), "corn_models")
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, model_filename)
                print(f"使用备用目录: {output_dir}")
            except Exception as e2:
                print(f"创建备用目录也失败: {str(e2)}")
                return False, best_val_acc
        
        try:
            # 使用二进制方式保存
            print(f"尝试保存模型到: {model_path}")
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            
            # 将缓冲区内容写入文件
            with open(model_path, 'wb') as f:
                f.write(buffer.getvalue())
            
            print(f"模型已成功保存至: {model_path}")
            
            # 如果是最佳模型，也保存一份
            if val_acc >= best_val_acc:
                best_model_path = os.path.join(output_dir, "improved_corn_model_best.pth")
                with open(best_model_path, 'wb') as f:
                    buffer.seek(0)
                    f.write(buffer.getvalue())
                print(f"最佳模型已保存至: {best_model_path}")
                return True, val_acc
            
            return False, best_val_acc
            
        except Exception as e:
            print(f"文件写入失败: {str(e)}")
            
            # 最后尝试使用pickle直接序列化
            try:
                import pickle
                pickle_path = os.path.join(output_dir, f"model_pickle_epoch{epoch}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"模型已使用pickle保存至: {pickle_path}")
            except Exception as e2:
                print(f"pickle保存也失败: {str(e2)}")
            
            return False, best_val_acc
        
    except Exception as e:
        print(f"保存模型过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, best_val_acc

def main(config):
    """主训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"\n改进模型配置:")
    print(f"  - 多尺度特征提取: {config['use_multiscale_features']}")
    print(f"  - 困难样本增强: {config['use_difficult_stage_enhancement']}")
    print(f"  - 对比学习: {config['use_contrastive_learning']}")
    print(f"  - 时间特征: {config['use_temporal_info']}")
    print(f"  - 注意力机制: {config['use_attention']}")
    
    # TODO: 数据加载和预处理
    # train_df = ...  # 加载训练数据
    # val_df = ...    # 加载验证数据
    
    # 转换
    train_transform = get_transforms(config, is_training=True)
    val_transform = get_transforms(config, is_training=False)
    
    # TODO: 创建数据集
    # train_dataset = CornDatasetBalanced(train_df, transform=train_transform, 
    #                                    include_time=config['use_temporal_info'],
    #                                    balanced_sampling=config['use_balanced_sampling'],
    #                                    class_weights_power=config['class_weights_power'])
    # val_dataset = CornDatasetBalanced(val_df, transform=val_transform, 
    #                                  include_time=config['use_temporal_info'])
    
    # TODO: 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
    #                           shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
    #                         shuffle=False, num_workers=4, pin_memory=True)
    
    # TODO: 计算类别权重
    # class_weights = calculate_class_weights(train_df['stage_idx'].values, 
    #                                        power=config['class_weights_power'])
    # class_weights = class_weights.to(device)
    
    # 创建模型
    model = ImprovedCornGrowthStageModel(
        num_classes=config['num_classes'],
        model_type=config['base_model'],
        pretrained=True,
        dropout_rate=config['dropout_rate'],
        include_time=config['use_temporal_info'],
        use_attention=config['use_attention'],
        use_contrastive=config['use_contrastive_learning']
    )
    model.to(device)
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['base_lr'], 
                            weight_decay=config['weight_decay'])
    
    # 创建学习率调度器
    # scheduler = get_lr_scheduler(optimizer, train_loader, config)
    
    # 创建损失函数 - 使用标签平滑
    if config.get('label_smoothing', 0) > 0:
        # 使用标签平滑交叉熵损失
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config['label_smoothing'],
            weight=None  # 可以传入class_weights
        )
        print(f"使用标签平滑交叉熵损失 (smoothing={config['label_smoothing']})")
    elif config['use_contrastive_learning']:
        # 使用组合损失函数（分类 + 对比学习）
        criterion = CombinedLossFunction(
            num_classes=config['num_classes'],
            use_focal=config['use_focal_loss'],
            gamma=config['gamma'],
            contrastive_weight=config['contrastive_weight'],
            class_weights=None  # 可以传入class_weights
        )
        print(f"使用组合损失函数 (分类 + 对比学习)")
        print(f"  - Focal Loss: {config['use_focal_loss']}")
        print(f"  - 对比学习权重: {config['contrastive_weight']}")
    else:
        # 使用传统损失函数
        if config['use_focal_loss']:
            criterion = FocalLoss(gamma=config['gamma'], alpha=None)  # 可以传入class_weights作为alpha
            print(f"使用Focal Loss (gamma={config['gamma']})")
        else:
            criterion = nn.CrossEntropyLoss()  # 可以传入weight=class_weights
            print(f"使用交叉熵损失")
    
    # 创建Mixup/Cutmix回调
    mixup_cutmix = None
    if config['use_mixup'] or config['use_cutmix']:
        mixup_cutmix = MixupCutmixCallback(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5
        )
    
    # 训练设置
    num_epochs = config['epochs']
    patience = config['patience']
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    # 输出目录
    output_dir = "models/improved_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # 阶段映射 (示例)
    stage_mapping = {
        '11': 0, '21': 1, '31': 2, '41': 3, '61': 4, 
        '71': 5, '81': 6, '91': 7, '99': 8
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        # train_loss, train_acc = train_one_epoch(
        #     model, train_loader, criterion, optimizer, device,
        #     scheduler=scheduler, use_amp=True,
        #     use_mixup=config['use_mixup'] or config['use_cutmix'],
        #     mixup_cutmix=mixup_cutmix
        # )
        
        # 验证
        # val_loss, val_acc, val_labels, val_preds = validate(
        #     model, val_loader, criterion, device, use_amp=True
        # )
        
        # print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存模型
        # is_best, best_val_acc = save_model(
        #     model, optimizer, epoch, config, stage_mapping,
        #     val_acc, val_loss, best_val_acc, output_dir
        # )
        
        # 早停
        # if is_best:
        #     no_improve_epochs = 0
        # else:
        #     no_improve_epochs += 1
        #     if no_improve_epochs >= patience:
        #         print(f"Early stopping after {epoch+1} epochs")
        #         break
    
    return model

if __name__ == "__main__":
    model = main(CONFIG)
    print("训练完成!") 

# 在文件末尾添加新的注意力机制实现
# ==================== 扩展注意力机制 ====================

class SEAttention(nn.Module):
    """SE - Squeeze-and-Excitation 注意力"""
    def __init__(self, in_channels, reduction=16):
        super(SEAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECAAttention(nn.Module):
    """ECA - Efficient Channel Attention"""
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        k_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        # 1D卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 激活
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SelfAttention(nn.Module):
    """Self-Attention - 自注意力机制"""
    def __init__(self, in_channels, reduction=8):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # 计算query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        
        # 计算注意力
        attention = torch.bmm(proj_query, proj_key)
        attention = self.softmax(attention)
        
        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # 残差连接
        out = self.gamma * out + x
        return out

class CoordAttention(nn.Module):
    """Coordinate Attention - 坐标注意力"""
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction)
        
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        return out

# 注意力机制工厂函数
def create_attention_module(attention_type, in_channels, **kwargs):
    """
    创建注意力模块的工厂函数
    
    Args:
        attention_type: 注意力类型 ('cbam', 'se', 'eca', 'self_attention', 'coord', 'vegetation', 'none')
        in_channels: 输入通道数
        **kwargs: 其他参数
        
    Returns:
        attention_module: 注意力模块实例
    """
    if attention_type == 'cbam':
        return CBAM(in_channels, kwargs.get('reduction', 16), kwargs.get('kernel_size', 7))
    elif attention_type == 'se':
        return SEAttention(in_channels, kwargs.get('reduction', 16))
    elif attention_type == 'eca':
        return ECAAttention(in_channels, kwargs.get('gamma', 2), kwargs.get('b', 1))
    elif attention_type == 'self_attention':
        return SelfAttention(in_channels, kwargs.get('reduction', 8))
    elif attention_type == 'coord':
        return CoordAttention(in_channels, kwargs.get('reduction', 32))
    elif attention_type == 'vegetation':
        return VegetationIndexAttention(in_channels)
    elif attention_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"未知的注意力类型: {attention_type}")

# 修改GrowthStageFeatureExtractor以支持不同注意力机制
class GrowthStageFeatureExtractorWithAttentionType(GrowthStageFeatureExtractor):
    """支持多种注意力机制的特征提取器"""
    def __init__(self, base_model_name='efficientnet_b3', pretrained=True, use_attention=True, 
                 use_multiscale=True, use_difficult_enhancement=True, attention_type='cbam'):
        # 临时禁用原有的注意力机制
        super(GrowthStageFeatureExtractorWithAttentionType, self).__init__(
            base_model_name=base_model_name, 
            pretrained=pretrained, 
            use_attention=False,  # 先禁用原有注意力
            use_multiscale=use_multiscale, 
            use_difficult_enhancement=use_difficult_enhancement
        )
        
        self.attention_type = attention_type
        self.use_attention = use_attention
        
        # 使用新的注意力机制
        if use_attention and attention_type != 'none':
            self.attention_module = create_attention_module(attention_type, self.enhanced_feature_dim)
            print(f"🎯 使用注意力机制: {attention_type}")
        else:
            self.attention_module = nn.Identity()
            print(f"❌ 不使用注意力机制")
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # 统一先提取主干特征
        if self.using_features_only:
            base_feat = self.base_model(x)[-1]
        else:
            base_feat = self.base_model(x)

        # 多尺度与主干融合（与父类一致，确保不空转）
        if self.use_multiscale:
            ms = self.multiscale_extractor(x)
            try:
                ms = F.interpolate(ms, size=base_feat.shape[-2:], mode='bilinear', align_corners=False, antialias=True)
            except TypeError:
                ms = F.interpolate(ms, size=base_feat.shape[-2:], mode='bilinear', align_corners=False)
            ms = self.ms_align_act(self.ms_align_bn(self.ms_align(ms)))
            fused = torch.cat([base_feat, ms], dim=1)
            features = self.fuse_act(self.fuse_bn(self.fuse_conv(fused)))
        else:
            features = base_feat
        
        # 困难样本增强
        if self.use_difficult_enhancement:
            # 细节增强
            enhanced_feat = self.detail_enhancer(features)
            
            # 全局平均池化用于困难样本判别
            global_feat = F.adaptive_avg_pool2d(enhanced_feat, (1, 1)).view(batch_size, -1)
            
            # 困难样本分类
            difficulty_logits = self.difficulty_classifier(global_feat)
            difficulty_probs = F.softmax(difficulty_logits, dim=1)
            
            # 根据困难程度选择特征增强器
            early_enhanced = self.difficult_stage_enhancer['early_stage'](enhanced_feat)
            late_enhanced = self.difficult_stage_enhancer['late_stage'](enhanced_feat)
            general_enhanced = self.difficult_stage_enhancer['general'](enhanced_feat)
            
            # 加权融合不同增强器的输出
            early_weight = difficulty_probs[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            late_weight = difficulty_probs[:, 1:2].unsqueeze(-1).unsqueeze(-1)
            general_weight = difficulty_probs[:, 2:3].unsqueeze(-1).unsqueeze(-1)
            
            adaptive_enhanced = (early_weight * early_enhanced + 
                               late_weight * late_enhanced + 
                               general_weight * general_enhanced)
            
            # 特征融合
            combined_feat = torch.cat([enhanced_feat, adaptive_enhanced], dim=1)
            features = self.feature_fusion(combined_feat)
        
        # 应用新的注意力机制
        if self.use_attention:
            features = self.attention_module(features)
        
        # 全局池化
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        if return_features:
            return flattened, {
                'raw_features': features,
                'attention_type': self.attention_type,
                'pooled_features': flattened
            }
        
        return flattened

# 修改ImprovedCornGrowthStageModel以支持attention_type参数
class ImprovedCornGrowthStageModelWithAttentionType(ImprovedCornGrowthStageModel):
    """支持多种注意力机制的玉米生育期识别模型"""
    
    def __init__(self, 
                 num_classes=9, 
                 model_type='efficientnet_b3', 
                 pretrained=True, 
                 dropout_rate=0.5, 
                 include_time=True,
                 use_attention=True,
                 use_contrastive=True,
                 use_multiscale=None,
                 use_difficult_enhancement=None,
                 attention_type='cbam'):  # 新增参数
        """
        初始化模型
        
        Args:
            attention_type: 注意力机制类型 ('cbam', 'se', 'eca', 'self_attention', 'coord', 'vegetation', 'none')
        """
        self.attention_type = attention_type
        
        # 调用父类初始化，但不直接调用super().__init__，而是手动实现以支持新的特征提取器
        nn.Module.__init__(self)
        
        self.model_type = model_type
        self.include_time = include_time
        self.use_attention = use_attention
        self.use_contrastive = use_contrastive
        
        # 参数优先级：传入参数 > CONFIG默认值
        self.use_multiscale = use_multiscale if use_multiscale is not None else CONFIG['use_multiscale_features']
        self.use_difficult_enhancement = use_difficult_enhancement if use_difficult_enhancement is not None else CONFIG['use_difficult_stage_enhancement']
        
        print(f"🔧 模型组件配置:")
        print(f"  ✅ 时间特征: {self.include_time}")
        print(f"  ✅ 注意力机制: {self.use_attention} ({attention_type})")
        print(f"  ✅ 多尺度特征: {self.use_multiscale}")
        print(f"  ✅ 困难样本增强: {self.use_difficult_enhancement}")
        print(f"  ✅ 对比学习: {self.use_contrastive}")
        
        # 使用新的特征提取器
        self.feature_extractor = GrowthStageFeatureExtractorWithAttentionType(
            base_model_name=model_type,
            pretrained=pretrained,
            use_attention=self.use_attention,
            use_multiscale=self.use_multiscale,
            use_difficult_enhancement=self.use_difficult_enhancement,
            attention_type=attention_type
        )
        
        self.feature_dim = self.feature_extractor.enhanced_feature_dim if self.use_difficult_enhancement else self.feature_extractor.feature_dim
        print(f"  📏 特征维度: {self.feature_dim}")
        
        # 时间特征处理
        if include_time:
            self.time_module = TimeFeatureModule(
                time_dim=3,
                hidden_dim=128,
                dropout_rate=dropout_rate
            )
            
            # 特征融合
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim + 128, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            
            final_dim = 512
            print(f"  🔗 融合后维度: {final_dim} (图像: {self.feature_dim} + 时间: 128)")
        else:
            final_dim = self.feature_dim
            print(f"  🖼️ 最终维度: {final_dim} (仅图像特征)")
        
        # 对比学习头
        if use_contrastive:
            self.contrastive_head = ContrastiveLearningHead(final_dim, 128)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
        
        # 随机深度模块
        use_stochastic_depth = CONFIG.get('use_stochastic_depth', False)
        if use_stochastic_depth:
            self.stochastic_depth = StochasticDepth(CONFIG.get('stochastic_depth_prob', 0.2))
        else:
            self.stochastic_depth = None
        
        # 初始化权重
        self._initialize_weights()

def get_available_attention_types():
    """获取可用的注意力机制类型"""
    return ['cbam', 'se', 'eca', 'self_attention', 'coord', 'vegetation', 'none']

def get_attention_info():
    """获取注意力机制信息"""
    return {
        'cbam': {
            'name': 'CBAM (Convolutional Block Attention Module)',
            'description': '结合通道注意力和空间注意力',
            'best_for': '通用图像特征增强'
        },
        'se': {
            'name': 'SE (Squeeze-and-Excitation)',
            'description': '通道级别的特征重校准',
            'best_for': '通道特征选择'
        },
        'eca': {
            'name': 'ECA (Efficient Channel Attention)',
            'description': '高效的通道注意力，避免降维',
            'best_for': '高效通道注意力'
        },
        'self_attention': {
            'name': 'Self-Attention',
            'description': '自注意力机制，捕获长距离依赖',
            'best_for': '全局特征关联'
        },
        'coord': {
            'name': 'Coordinate Attention',
            'description': '坐标注意力，保留位置信息',
            'best_for': '位置敏感任务'
        },
        'vegetation': {
            'name': 'Vegetation Index Attention',
            'description': '植被指数注意力，增强绿色植被特征',
            'best_for': '农业图像分析'
        },
        'none': {
            'name': 'No Attention',
            'description': '不使用注意力机制',
            'best_for': '基线对比'
        }
    } 