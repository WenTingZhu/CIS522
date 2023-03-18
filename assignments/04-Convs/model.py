import torch
import torch.nn as nn

# import torch.nn.functional as F
torch.manual_seed(12321)


class Model(torch.nn.Module):
    """
    A simple CNN model that reaches 0.55 accuracy that cost less than 15 sec for an epoch, with batch size 200
    and learning rate 6e-3 on CIFAR10 dataset
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Structure of the model that runs as fast as possible
        Args:
            num_channels (int): number of channels in the input image
            num_classes (int): number of classes in the dataset
        """
        super(Model, self).__init__()

        # self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ker = 5
        self.pad = (self.ker - 1) // 2
        self.nchan = 16
        self.conv2 = nn.Conv2d(
            num_channels,
            16,
            kernel_size=3,
            stride=2,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv1 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, num_classes)
        # self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        # self.fc2 = nn.Linear(256, 100)
        # self.fc3 = nn.Linear(100, num_classes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # self.convdepth = nn.Conv2d(num_channels,self.nchan,kernel_size = 1)
        # nn.init.xavier_uniform_(self.convdepth.weight)
        self.model = nn.Sequential(
            self.conv2,
            nn.ReLU(),
            self.bn2,
            self.pool2,
            # self.conv1,
            # nn.ReLU(),
            # self.bn2,
            # self.pool2,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=6e-3)
        # criterion = nn.CrossEntropyLoss()
        # for _ in range(20):
        #     x = torch.randn(200, 3, 32, 32)
        #     y = torch.randn(200, 10)
        #     y_pred = self.model(x)
        #     loss = criterion(y_pred, y)
        #     loss.backward()
        #     # optimizer.step()
        #     optimizer.zero_grad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        Args:
            x (torch.Tensor): input image
        Returns:
            torch.Tensor: output logits
        """
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.pool1(x)

        # x = self.conv2(x)
        # # x = self.convdepth(x)
        # x = F.relu(x)
        # x = self.pool2(x)

        # x = x.view(-1, self.nchan * 8 * 8)
        # # x = F.relu(self.fc1(x))
        # # x = F.relu(self.fc2(x))
        # x = self.fc1(x)
        x = self.model(x)

        return x


# def label_smoothing_loss(inputs, targets, alpha):
#     log_probs = torch.nn.functional.log_softmax(inputs, dim=1, _stacklevel=5)
#     kl = -log_probs.mean(dim=1)
#     xent = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
#     loss = (1 - alpha) * xent + alpha * kl
#     return loss


# class GhostBatchNorm(nn.BatchNorm2d):
#     def __init__(self, num_features, num_splits, **kw):
#         super().__init__(num_features, **kw)

#         running_mean = torch.zeros(num_features * num_splits)
#         running_var = torch.ones(num_features * num_splits)

#         self.weight.requires_grad = False
#         self.num_splits = num_splits
#         self.register_buffer("running_mean", running_mean)
#         self.register_buffer("running_var", running_var)

#     def train(self, mode=True):
#         if (self.training is True) and (mode is False):
#             # lazily collate stats when we are going to use them
#             self.running_mean = torch.mean(
#                 self.running_mean.view(self.num_splits, self.num_features), dim=0
#             ).repeat(self.num_splits)
#             self.running_var = torch.mean(
#                 self.running_var.view(self.num_splits, self.num_features), dim=0
#             ).repeat(self.num_splits)
#         return super().train(mode)

#     def forward(self, input):
#         n, c, h, w = input.shape
#         if self.training or not self.track_running_stats:
#             assert (
#                 n % self.num_splits == 0
#             ), f"Batch size ({n}) must be divisible by num_splits ({self.num_splits}) of GhostBatchNorm"
#             return F.batch_norm(
#                 input.view(-1, c * self.num_splits, h, w),
#                 self.running_mean,
#                 self.running_var,
#                 self.weight.repeat(self.num_splits),
#                 self.bias.repeat(self.num_splits),
#                 True,
#                 self.momentum,
#                 self.eps,
#             ).view(n, c, h, w)
#         else:
#             return F.batch_norm(
#                 input,
#                 self.running_mean[: self.num_features],
#                 self.running_var[: self.num_features],
#                 self.weight,
#                 self.bias,
#                 False,
#                 self.momentum,
#                 self.eps,
#             )


# def conv_bn_relu(c_in, c_out, kernel_size=(3, 3), padding=(1, 1)):
#     return nn.Sequential(
#         nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=False),
#         GhostBatchNorm(c_out, num_splits=16),
#         nn.CELU(alpha=0.3),
#     )


# def conv_pool_norm_act(c_in, c_out):
#     return nn.Sequential(
#         nn.Conv2d(c_in, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         GhostBatchNorm(c_out, num_splits=16),
#         nn.CELU(alpha=0.3),
#     )


# def patch_whitening(data, patch_size=(3, 3)):
#     # Compute weights from data such that
#     # torch.std(F.conv2d(data, weights), dim=(2, 3))
#     # is close to 1.
#     h, w = patch_size
#     c = data.size(1)
#     patches = data.unfold(2, h, 1).unfold(3, w, 1)
#     patches = patches.transpose(1, 3).reshape(-1, c, h, w).to(torch.float32)

#     n, c, h, w = patches.shape
#     X = patches.reshape(n, c * h * w)
#     X = X / (X.size(0) - 1) ** 0.5
#     covariance = X.t() @ X

#     eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

#     eigenvalues = eigenvalues.flip(0)

#     eigenvectors = eigenvectors.t().reshape(c * h * w, c, h, w).flip(0)

#     return eigenvectors / torch.sqrt(eigenvalues + 1e-2).view(-1, 1, 1, 1)


# class Model(nn.Module):
#     def __init__(self, num_channels: int, num_classes: int):
#         super().__init__()

#         c = 10000

#         conv1 = nn.Conv2d(
#             num_channels, c, kernel_size=(3, 3), padding=(1, 1), bias=False
#         )
#         # conv1.weight.data = []
#         # conv1.weight.requires_grad = False

#         self.conv1 = conv1
#         self.conv2 = conv_bn_relu(c, 64, kernel_size=(1, 1), padding=0)
#         self.conv3 = conv_pool_norm_act(64, 128)
#         self.conv4 = conv_bn_relu(128, 128)
#         self.conv5 = conv_bn_relu(128, 128)
#         self.conv6 = conv_pool_norm_act(128, 256)
#         self.conv7 = conv_pool_norm_act(256, 512)
#         self.conv8 = conv_bn_relu(512, 512)
#         self.conv9 = conv_bn_relu(512, 512)
#         self.pool10 = nn.MaxPool2d(kernel_size=4, stride=4)
#         self.linear11 = nn.Linear(512, num_classes, bias=False)
#         self.scale_out = 0.125

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x + self.conv5(self.conv4(x))
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = x + self.conv9(self.conv8(x))
#         x = self.pool10(x)
#         x = x.reshape(x.size(0), x.size(1))
#         x = self.linear11(x)
#         x = self.scale_out * x
#         return x
