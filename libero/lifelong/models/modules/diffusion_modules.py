import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for time steps in diffusion models."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DenoisingNetwork(nn.Module):
    """
    Vanilla noise prediction MLP for DDPM, taking in noisy action, obs features, task features, and time embedding.
    """
    def __init__(self, action_dim, obs_feat_dim, task_feat_dim, time_emb_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        self.obs_feat_dim = obs_feat_dim
        self.task_feat_dim = task_feat_dim
        self.time_emb_dim = time_emb_dim
        input_dim = action_dim + obs_feat_dim + task_feat_dim + time_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, obs_feat, task_feat, time_emb):
        """
        x: noise action (B, action_dim)
        obs_feat: state features (B, obs_feat_dim)
        task_feat: task features (B, task_feat_dim)
        time_emb: time step embedding (B, time_emb_dim)
        """
        inp = torch.cat([x, obs_feat, task_feat, time_emb], dim=-1)
        return self.net(inp)


class DDPM(nn.Module):
    """
    Vanilla Denoising Diffusion Probabilistic Model (DDPM) for action generation, with a noise prediction network.
    """
    def __init__(self, noise_pred_net, beta_start=0.0001, beta_end=0.02, num_diffusion_steps=100):
        super().__init__()
        self.noise_pred_net = noise_pred_net
        self.num_diffusion_steps = num_diffusion_steps

        # 定义 beta 调度（线性）
        self.register_buffer(
            'betas', torch.linspace(beta_start, beta_end, num_diffusion_steps)
        )
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x0, t, noise):
        """
        forwarding noise: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

    def p_losses(self, x0, obs_feat, task_feat, t):
        """
        diffusion loss: MSE between predicted noise and true noise
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        time_emb = self.get_time_emb(t)
        pred_noise = self.noise_pred_net(x_t, obs_feat, task_feat, time_emb)
        return F.mse_loss(pred_noise, noise)

    def get_time_emb(self, t):
        """
        获取时间步嵌入，这里假设 noise_pred_net 内部有 time_emb 模块，或者外部传入。
        为简化，我们要求 noise_pred_net 接收 time_emb 作为输入，这里生成 time_emb。
        """
        # 实际使用中，time_emb 由 SinusoidalPosEmb 生成，我们在外部处理。
        # 此方法仅用于示例，实际会在训练时调用外部嵌入模块。
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, obs_feat, task_feat, time_emb_fn, shape, device):
        """
        从纯噪声开始，逐步去噪生成样本。
        time_emb_fn: 函数，输入 t (B,) 返回 time_emb (B, time_emb_dim)
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_diffusion_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            time_emb = time_emb_fn(t)
            pred_noise = self.noise_pred_net(x, obs_feat, task_feat, time_emb)
            alpha = self.alphas[t].view(-1, 1)
            alpha_bar = self.alpha_bars[t].view(-1, 1)
            beta = self.betas[t].view(-1, 1)

            # 根据 DDPM 公式更新
            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_bar).sqrt() * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                x = x + beta.sqrt() * noise
        return x