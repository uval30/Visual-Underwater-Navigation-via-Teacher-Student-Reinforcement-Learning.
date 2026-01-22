import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """Simple CNN encoder for Isaac camera images."""
    def __init__(self, in_channels: int = 1, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))

class StudentGaussianPolicy(nn.Module):
    """
    Image encoder + orientation -> Gaussian action policy + Teacher state prediction.
    """
    LOG_STD_MIN_ACTION, LOG_STD_MAX_ACTION = -4.0, 1.0
    LOG_STD_MIN_STATE, LOG_STD_MAX_STATE = -4.0, -1

    def __init__(self, action_dim, env, in_channels=1, latent_dim_img=128, ori_dim=4):
        super().__init__()
        self.encoder = ImageEncoder(in_channels=in_channels, latent_dim=latent_dim_img)
        feat_dim = latent_dim_img + ori_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
        )

        state_dim = env.observation_space.shape[0]
        # Aux heads for Teacher State Prediction (Auxiliary Task)
        self.teacher_state_mu = nn.Linear(feat_dim, state_dim)
        self.teacher_state_log_std = nn.Linear(feat_dim, state_dim)

        # Action heads
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, images, ori):
        z = torch.cat([self.encoder(images), ori], dim=1)
        h = self.backbone(z)

        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN_ACTION, self.LOG_STD_MAX_ACTION)

        t_mu = self.teacher_state_mu(z)
        t_log_std = torch.clamp(self.teacher_state_log_std(z), self.LOG_STD_MIN_STATE, self.LOG_STD_MAX_STATE)

        return mu, log_std, t_mu, t_log_std

    def act(self, images, ori, deterministic=False):
        mu, log_std, _, _ = self(images, ori)
        return mu if deterministic else mu + torch.randn_like(mu) * log_std.exp()
