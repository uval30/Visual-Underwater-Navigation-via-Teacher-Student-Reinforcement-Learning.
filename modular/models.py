import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """
    Simple CNN encoder for Isaac camera images.
    Input:  (N, C, H, W)
    Output: (N, latent_dim)
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> (N, 64, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                   # -> (N, 64*4*4 = 1024)
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        z = self.fc(x)
        return z


class StudentGaussianPolicy(nn.Module):
    """
    Image encoder + orientation -> Gaussian action policy.
    Outputs:
      - mu, log_std     : action distribution (what SAC cares about)
      - teacher_mu, teacher_log_std : predicted state distribution (only you use)
    """

    LOG_STD_MIN_ACTION = -4.0   # std ≈ 0.018
    LOG_STD_MAX_ACTION =  1.0   # std ≈ 7.39  (or even 3.0 → std ≈ 20)

    LOG_STD_MIN_STATE  = -4.0
    LOG_STD_MAX_STATE  =  -1   # std ≈ 1.0 is safely above your 0.3

    def __init__(self, action_dim, env, in_channels=1,
                 latent_dim_img=128, ori_dim=4):
        super().__init__()

        self.encoder = ImageEncoder(in_channels=in_channels,
                                    latent_dim=latent_dim_img)

        feat_dim = latent_dim_img + ori_dim   # this must match SB3 features_dim

        # == this is what SB3's actor.latent_pi mimics ==
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        state_dim = env.observation_space.shape[0]

        # extra heads for teacher-state prediction (BC loss only)
        self.teacher_state_mu  = nn.Linear(feat_dim, state_dim)
        self.teacher_state_log_std = nn.Linear(feat_dim, state_dim)

        # == these are what SB3's actor.mu / actor.log_std mimic ==
        self.mu_head      = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, images, ori):
        """
        images: (N, C, H, W)
        ori:    (N, 4)
        Returns:
            mu:              (N, act_dim)
            log_std:         (N, act_dim)
            teacher_mu:      (N, state_dim)
            teacher_log_std: (N, state_dim)
        """
        z_img = self.encoder(images)          # (N, latent_dim_img)
        z = torch.cat([z_img, ori], dim=1)       # (N, feat_dim)

        h = self.backbone(z)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN_ACTION, self.LOG_STD_MAX_ACTION)

        teacher_mu = self.teacher_state_mu(z)
        teacher_log_std = torch.clamp(
            self.teacher_state_log_std(z),
            self.LOG_STD_MIN_STATE,
            self.LOG_STD_MAX_STATE,
        )

        return mu, log_std, teacher_mu, teacher_log_std

    def act(self, images, ori, deterministic=False):
        mu, log_std, _, _ = self(images, ori)
        if deterministic:
            return mu

        std = log_std.exp()
        eps = torch.randn_like(mu)
        return mu + eps * std
