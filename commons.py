"""
 > Common/standard network archutectures and modules
 > Credit for some functions
    * github.com/eriklindernoren/PyTorch-GAN
    * pluralsight.com/guides/artistic-neural-style-transfer-with-pytorch
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torchvision.models import vgg19


def Weights_Normal(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and m.weight is not None:
        # 处理 Conv2dNormActivation 等复合结构
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        except:
            pass



class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


# ⭐⭐⭐ 新增：针对完整模型的增强损失函数 ⭐⭐⭐

class SSIMLoss(nn.Module):
    """结构相似度损失"""

    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        return 1 - self._ssim(img1, img2)

    def _ssim(self, img1, img2):
        # 简化的SSIM实现
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()



class EdgeAwareLoss(nn.Module):
    """边缘感知损失"""

    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        # Sobel 滤波器 - 注册为缓冲区，这样会自动移动到正确设备
        self.register_buffer('sobel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))



    def sobel_filter(self, x):
        # 确保权重在正确设备上
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)

        # 扩展 Sobel 滤波器到与输入相同的通道数
        weight_x = sobel_x.expand(x.size(1), 1, 3, 3)
        weight_y = sobel_y.expand(x.size(1), 1, 3, 3)

        edge_x = F.conv2d(x, weight_x, padding=1, groups=x.size(1))
        edge_y = F.conv2d(x, weight_y, padding=1, groups=x.size(1))

        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge

    def forward(self, generated, target):
        edge_gen = self.sobel_filter(generated)
        edge_target = self.sobel_filter(target)
        return F.l1_loss(edge_gen, edge_target)

class ColorConsistencyLoss(nn.Module):
     # """色彩一致性损失（优化版）- 重点抑制红色偏色"""

    def __init__(self, red_weight=1.5, green_weight=1.0, blue_weight=1.0):
        super(ColorConsistencyLoss, self).__init__()
        # 为不同通道设置权重：红色通道权重更高，优先约束红色偏移
        self.channel_weights = torch.tensor([red_weight, green_weight, blue_weight])

    def forward(self, generated, target):
        # IrdUgan. 计算 RGB 三通道的均值（维度：[B, 3]，B=批次大小，3=RGB）
        # generated/target 形状需为 [B, 3, H, W]（PyTorch 标准图像格式）
        gen_mean = torch.mean(generated, dim=[2, 3])  # 对 H、W 维度求均值，得到 [B, 3]
        target_mean = torch.mean(target, dim=[2, 3])

        # 2. 移动权重到生成图的设备（避免 CPU/GPU 不匹配）
        self.channel_weights = self.channel_weights.to(generated.device)

        # 3. 计算带权重的通道均值差异（红色通道差异乘以更高权重）
        mean_diff = torch.abs(gen_mean - target_mean)  # [B, 3]：每个通道的绝对差异
        weighted_diff = mean_diff * self.channel_weights  # [B, 3]：红色通道差异被放大

        # 4. 返回平均损失（对批次和通道求平均）
        return torch.mean(weighted_diff)


class EnhancedGeneratorLoss(nn.Module):
    """针对完整模型的增强生成器损失"""

    def __init__(self, lambda_gan=1.0, lambda_l1=7.0, lambda_vgg=3.0,
                 lambda_ssim=2.0, lambda_edge=1.0, lambda_color=0.5):
        super(EnhancedGeneratorLoss, self).__init__()

        # 基础损失
        self.l1_loss = nn.L1Loss()

        # ⭐⭐⭐ 新增损失 ⭐⭐⭐
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeAwareLoss()
        self.color_loss = ColorConsistencyLoss()

        # 损失权重
        self.lambda_gan = lambda_gan
        self.lambda_l1 = lambda_l1
        self.lambda_vgg = lambda_vgg
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_color = lambda_color

    def forward(self, generated, target, vgg_loss_func=None):
        """
        Args:
            generated: 生成图像
            target: 真实图像
            vgg_loss_func: VGG感知损失函数，需要从外部传入
        """
        # 确保输入在相同设备上
        if generated.device != target.device:
            target = target.to(generated.device)

        # ⭐⭐⭐ 修复：确保边缘损失在正确设备上 - 修复StopIteration错误 ⭐⭐⭐
        current_device = generated.device

        # 移动边缘损失到正确设备
        try:
            # 尝试获取边缘损失的参数设备
            edge_device = next(self.edge_loss.parameters()).device
            if edge_device != current_device:
                self.edge_loss = self.edge_loss.to(current_device)
        except StopIteration:
            # 如果EdgeLoss没有参数，直接移动到设备
            self.edge_loss = self.edge_loss.to(current_device)

        # 确保VGG损失函数在正确设备上
        if vgg_loss_func is not None:
            try:
                vgg_device = next(vgg_loss_func.parameters()).device
                if vgg_device != current_device:
                    vgg_loss_func = vgg_loss_func.to(current_device)
            except StopIteration:
                vgg_loss_func = vgg_loss_func.to(current_device)

        total_loss = 0.0
        loss_dict = {}

        # 基础损失
        loss_l1 = self.l1_loss(generated, target)
        total_loss += self.lambda_l1 * loss_l1
        loss_dict['l1'] = loss_l1.item()

        # VGG感知损失（如果提供了VGG损失函数）
        if vgg_loss_func is not None:
            loss_vgg = vgg_loss_func(generated, target)
            total_loss += self.lambda_vgg * loss_vgg
            loss_dict['vgg'] = loss_vgg.item()
        else:
            loss_dict['vgg'] = 0.0

        # ⭐⭐⭐ 新增损失 ⭐⭐⭐
        loss_ssim = self.ssim_loss(generated, target)
        total_loss += self.lambda_ssim * (1 - loss_ssim)  # SSIM越大越好，所以用1-SSIM
        loss_dict['ssim'] = loss_ssim.item()

        loss_edge = self.edge_loss(generated, target)
        total_loss += self.lambda_edge * loss_edge
        loss_dict['edge'] = loss_edge.item()

        loss_color = self.color_loss(generated, target)
        total_loss += self.lambda_color * loss_color
        loss_dict['color'] = loss_color.item()

        loss_dict['total_content'] = total_loss.item()

        return total_loss, loss_dict


class VGG19ForPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGG19ForPerceptualLoss, self).__init__()
        # 修复：使用新的weights参数替代弃用的pretrained参数
        vgg = vgg19(weights='DEFAULT').features  # 或者使用 weights='IMAGENET1K_V1'

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # 分层提取特征
        for x in range(4):  # conv1
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  # conv2
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18):  # conv3, conv4
            self.slice3.add_module(str(x), vgg[x])
        for x in range(18, 27):  # conv5
            self.slice4.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """计算x和y在VGG特征空间上的感知损失"""
        # 归一化到VGG的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x + 1) / 2  # 从[-1,1]到[0,1]
        x = (x - mean) / std
        y = (y + 1) / 2
        y = (y - mean) / std

        # 提取多层特征并计算损失
        h_x = self.slice1(x)
        h_y = self.slice1(y)
        h1_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        h2_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        h3_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        h4_loss = F.l1_loss(h_x, h_y)

        return h1_loss + h2_loss + h3_loss + h4_loss


class EnhancedLoss(nn.Module):
    """
    修改3: 创建组合损失函数
    整合相对论GAN损失、L1损失、感知损失和梯度损失
    """

    def __init__(self, lambda_l1=100.0, lambda_perceptual=1.0, lambda_gradient=1.0, device='cuda'):
        super(EnhancedLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gradient = lambda_gradient

        # L1 损失
        self.l1_loss = nn.L1Loss()

        # 感知损失
        self.vgg_loss = VGG19ForPerceptualLoss().to(device)

        # Sobel梯度算子
        self.sobel_x, self.sobel_y = self._get_sobel_kernels(device)

    def _get_sobel_kernels(self, device):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        return sobel_x, sobel_y

    def gradient_loss(self, pred, target):
        """计算梯度损失 - 修复版本"""
        # 扩展Sobel核以匹配输入通道数
        sobel_x = self.sobel_x.repeat(pred.shape[1], 1, 1, 1)  # [3, 1, 3, 3]
        sobel_y = self.sobel_y.repeat(pred.shape[1], 1, 1, 1)  # [3, 1, 3, 3]

        # 预测图像的梯度
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])

        # 真实图像的梯度
        grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

        # 计算L1损失
        loss = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)
        return loss

    def relativistic_gan_loss(self, d_real, d_fake, for_generator=True):
        """计算相对论GAN损失"""
        if for_generator:
            # 生成器损失：让假图像看起来比真图像更真实
            real_loss = F.binary_cross_entropy_with_logits(
                d_real - d_fake.mean(0, keepdim=True), torch.zeros_like(d_real))
            fake_loss = F.binary_cross_entropy_with_logits(
                d_fake - d_real.mean(0, keepdim=True), torch.ones_like(d_fake))
        else:
            # 判别器损失：让真图像看起来比假图像更真实
            real_loss = F.binary_cross_entropy_with_logits(
                d_real - d_fake.mean(0, keepdim=True), torch.ones_like(d_real))
            fake_loss = F.binary_cross_entropy_with_logits(
                d_fake - d_real.mean(0, keepdim=True), torch.zeros_like(d_fake))

        return (real_loss + fake_loss) / 2

    def forward(self, discriminator, gen_output, target, real_input, fake_input, for_generator=True):
        """
        计算组合损失

        Args:
            discriminator: 判别器
            gen_output: 生成器输出
            target: 真实目标
            real_input: 判别器的真实输入
            fake_input: 判别器的假输入
            for_generator: 是否为生成器计算损失
        """
        total_loss = 0
        losses = {}

        # 1. 相对论GAN损失
        d_real = discriminator(real_input, target)
        d_fake = discriminator(fake_input, target)

        gan_loss = self.relativistic_gan_loss(d_real, d_fake, for_generator)
        total_loss += gan_loss
        losses['GAN'] = gan_loss

        if for_generator:
            # 2. L1损失
            l1_l = self.l1_loss(gen_output, target) * self.lambda_l1
            total_loss += l1_l
            losses['L1'] = l1_l

            # 3. 感知损失
            if self.lambda_perceptual > 0:
                perc_l = self.vgg_loss(gen_output, target) * self.lambda_perceptual
                total_loss += perc_l
                losses['Perceptual'] = perc_l

            # 4. 梯度损失
            if self.lambda_gradient > 0:
                grad_l = self.gradient_loss(gen_output, target) * self.lambda_gradient
                total_loss += grad_l
                losses['Gradient'] = grad_l

            losses['Total'] = total_loss

        return total_loss, losses