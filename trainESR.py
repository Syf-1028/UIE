"""
 > Two-stage training pipeline for Enhanced FUnIE-GAN
 > Stage 1: Basic training with GAN + L1 losses only
 > Stage 2: Refined training with all losses (GAN + L1 + Perceptual + Gradient)
"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, EnhancedLoss
from nets.ESR import GeneratorFunieGAN, RelativisticDiscriminator
from utils.data_utils import GetTrainingPairs, GetValImage


def gradient_penalty(discriminator, real_imgs, fake_imgs, distorted_imgs):
    """梯度惩罚 - 防止判别器过强"""
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(real_imgs.device)

    # 插值样本
    interpolated = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolated = discriminator(interpolated, distorted_imgs)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


## 配置默认参数
def get_default_args():
    """提供默认训练参数"""
    args = argparse.Namespace()
    args.cfg_file = "D:/PycharmProjects/FUnIE-GAN-master/PyTorch/configs/train_ufo.yaml"
    args.epoch = 0
    args.num_epochs = 161  # 修改：总轮数改为150
    args.batch_size = 8
    args.lr = 0.0003
    args.b1 = 0.5
    args.b2 = 0.99
    # 两阶段参数
    args.stage1_epochs = 80    # 第一阶段轮数
    args.stage2_epochs = 81    # 第二阶段轮数
    # 损失权重参数
    args.lambda_l1_stage1 = 100.0
    args.lambda_perceptual_stage1 = 0.0    # 第一阶段关闭
    args.lambda_gradient_stage1 = 0.0      # 第一阶段关闭

    args.lambda_l1_stage2 = 15.0
    args.lambda_perceptual_stage2 = 1.0    # 第二阶段开启
    args.lambda_gradient_stage2 = 1.0      # 第二阶段开启
    return args

## 解析参数
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="D:/PycharmProjects/FUnIE-GAN-master/PyTorch/configs/train_ufo.yaml")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=161)  # 修改
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.99)
    # 两阶段参数
    parser.add_argument("--stage1_epochs", type=int, default=80)
    parser.add_argument("--stage2_epochs", type=int, default=81)
    # 损失权重
    parser.add_argument("--lambda_l1_stage1", type=float, default=100.0)
    parser.add_argument("--lambda_perceptual_stage1", type=float, default=0.0)
    parser.add_argument("--lambda_gradient_stage1", type=float, default=0.0)
    parser.add_argument("--lambda_l1_stage2", type=float, default=15.0)
    parser.add_argument("--lambda_perceptual_stage2", type=float, default=1.0)
    parser.add_argument("--lambda_gradient_stage2", type=float, default=1.0)
    args = parser.parse_args()
else:
    args = get_default_args()

## 训练参数
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2

# 加载配置文件
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]

## 创建保存目录
samples_dir = os.path.join("samples/two_stageL1/", dataset_name)  # 修改目录名
checkpoint_dir = os.path.join("checkpoints/two_stageL1/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

## 初始化模型
generator = GeneratorFunieGAN()
discriminator = RelativisticDiscriminator()

# 设备配置
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Tensor = torch.cuda.FloatTensor
    device = 'cuda'
else:
    Tensor = torch.FloatTensor
    device = 'cpu'

# 加载权重或初始化
if args.epoch != 0:
    generator.load_state_dict(torch.load(f"D:/PycharmProjects/FUnIE-GAN-master/PyTorch/checkpoints/two_stageL1/{dataset_name}/generator_{args.epoch}.pth"))
    discriminator.load_state_dict(torch.load(f"D:/PycharmProjects/FUnIE-GAN-master/PyTorch/checkpoints/two_stageL1/{dataset_name}/discriminator_{args.epoch}.pth"))
    print(f"Loaded model from epoch {args.epoch}")
else:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)

# 修改优化器配置 - 判别器使用更小的学习率
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate * 0.1, betas=(lr_b1, lr_b2))  # 判别器学习率降低10倍

## 数据加载
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

## 训练循环 - 两阶段训练
for epoch in range(epoch, num_epochs):

    # 确定当前阶段和损失函数
    if epoch < args.stage1_epochs:
        # 第一阶段：基础训练 (只使用GAN + L1损失)
        current_loss = EnhancedLoss(
            lambda_l1=args.lambda_l1_stage1,
            lambda_perceptual=args.lambda_perceptual_stage1,  # 0.0
            lambda_gradient=args.lambda_gradient_stage1,      # 0.0
            device=device
        )
        stage_name = "Stage1-Basic"
        stage_info = f"L1:{args.lambda_l1_stage1}, Perc:0, Grad:0"

    else:
        # 第二阶段：精细调优 (使用所有损失)
        current_loss = EnhancedLoss(
            lambda_l1=args.lambda_l1_stage2,
            lambda_perceptual=args.lambda_perceptual_stage2,  # 1.0
            lambda_gradient=args.lambda_gradient_stage2,      # 1.0
            device=device
        )
        stage_name = "Stage2-Refine"
        stage_info = f"L1:{args.lambda_l1_stage2}, Perc:{args.lambda_perceptual_stage2}, Grad:{args.lambda_gradient_stage2}"

    # 阶段切换提示
    if epoch == args.stage1_epochs:
        print(f"\n{'='*50}")
        print(f"Switching to Stage 2: Refined Training")
        print(f"Loss weights: {stage_info}")
        print(f"{'='*50}\n")

    for i, batch in enumerate(dataloader):
        # 输入数据
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        ## 训练生成器
        ## 训练生成器 - 简化版本
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)

        # 使用EnhancedLoss计算生成器损失（不包含梯度惩罚）
        loss_G, g_losses = current_loss(
            discriminator=discriminator,
            gen_output=imgs_fake,
            target=imgs_good_gt,
            real_input=imgs_good_gt,
            fake_input=imgs_fake,
            for_generator=True
        )

        loss_G.backward()
        optimizer_G.step()

        ## 训练判别器 - 修复版本
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)

        # 计算梯度惩罚
        gp_loss = gradient_penalty(discriminator, imgs_good_gt, imgs_fake.detach(), imgs_distorted)
        gp_weight = 10.0  # 梯度惩罚权重

        # 计算相对论损失
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        pred_fake = discriminator(imgs_fake.detach(), imgs_distorted)

        # 诊断信息
        with torch.no_grad():
            d_real_mean = pred_real.mean().item()
            d_fake_mean = pred_fake.mean().item()
            d_real_std = pred_real.std().item()
            d_fake_std = pred_fake.std().item()

        # 相对论GAN损失
        loss_real = F.binary_cross_entropy_with_logits(
            pred_real - pred_fake.mean(0, keepdim=True),
            torch.ones_like(pred_real)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            pred_fake - pred_real.mean(0, keepdim=True),
            torch.zeros_like(pred_fake)
        )

        # 总损失 = 相对论损失 + 梯度惩罚
        loss_D = (loss_real + loss_fake) / 2 + gp_weight * gp_loss

        # 极端值检测和修复
        if abs(d_real_mean) > 5 or abs(d_fake_mean) > 5:
            print(f"\n[警告] 判别器输出极端值: d_real={d_real_mean:.1f}, d_fake={d_fake_mean:.1f}")
            # 增加梯度惩罚权重
            loss_D = loss_D + gp_weight * 2 * gp_loss

        loss_D.backward()

        # 检查梯度
        d_grad_norm = 0.0
        for param in discriminator.parameters():
            if param.grad is not None:
                d_grad_norm += param.grad.norm().item()

        optimizer_D.step()

        ## 打印日志 - 显示当前阶段信息
        if not i % 50:
            # 构建详细的损失信息
            loss_details = []
            for loss_name, loss_val in g_losses.items():
                if loss_name != 'Total':  # 不显示总损失，避免重复
                    loss_details.append(f"{loss_name}:{loss_val:.3f}")

            loss_str = ", ".join(loss_details)

            sys.stdout.write(f"\r[{stage_name} Epoch {epoch}/{num_epochs}] "
                             f"Batch {i}/{len(dataloader)} | "
                             f"D:{loss_D.item():.3f}(gp:{gp_loss.item():.3f}, grad:{d_grad_norm:.1f}), "
                             f"G:{loss_G.item():.3f} | "
                             f"d_real:{d_real_mean:.3f}±{d_real_std:.2f}, "
                             f"d_fake:{d_fake_mean:.3f}±{d_fake_std:.2f} | "
                             f"{loss_str}")

        ## 保存验证样本
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, f"samples/two_stageL1/{dataset_name}/{batches_done}.png", nrow=5, normalize=True)

    ## 保存模型 checkpoint - 特别保存阶段切换点
    if epoch % ckpt_interval == 0:
        torch.save(generator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/generator_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/discriminator_{epoch}.pth")

    # 特别保存阶段结束时的模型
    if epoch == args.stage1_epochs - 1:  # 第一阶段最后一代
        torch.save(generator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/generator_stage1_final.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/discriminator_stage1_final.pth")
        print(f"\nSaved Stage 1 final models at epoch {epoch}")

## 训练完成保存最终模型
torch.save(generator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/generator_final.pth")
torch.save(discriminator.state_dict(), f"checkpoints/two_stageL1/{dataset_name}/discriminator_final.pth")
print(f"\nTraining completed! Final models saved.")