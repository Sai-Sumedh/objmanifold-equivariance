import torch
import torch.nn as nn
import torch.nn.functional as F

# credit to E. Theodosis for the GCNN layer (paper cited below)
# Theodosis, E., Helwani, K. and Ba, D., 2023. Learning Linear Groups in Neural Networks. arXiv preprint arXiv:2305.18552.

#Specify device 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# PYTORCH_ENABLE_MPS_FALLBACK=1

def sinc_int(x, A, device, delta=1):
    """
    x: (N, C, H, W)
    A: (N, C, 2, 2)
    """
    bs = x.shape
    os = torch.Size([x.shape[2], x.shape[3]])
    ns = torch.Size([int(s * delta) for s in os])
    new_x = torch.zeros((bs[0], bs[1], ns[0], ns[1]), device=device)

    center_x = torch.arange(ns[0], device=device) - ns[0] // 2
    center_y = torch.arange(ns[1], device=device) - ns[1] // 2
    centered = torch.stack(torch.meshgrid(center_x, center_y), -1).reshape(-1, 2).T
    centered = centered.float().reshape(1, 1, 2, -1)
    transform = A @ centered
    transform_x = transform[:, :, 0, :] + ns[0] // 2 - 1
    transform_y = transform[:, :, 1, :] + ns[1] // 2
    old_x = torch.arange(os[0], device=device)
    old_y = torch.arange(os[1], device=device)

    tile_x = torch.tile(
        transform_x[:, :, None, :], (1, 1, os[0], 1)
    ) - delta * torch.tile(old_x[None, None, :, None], (bs[0], bs[1], 1, ns[0] * ns[1]))
    tile_y = torch.tile(
        transform_y[:, :, None, :], (1, 1, os[1], 1)
    ) - delta * torch.tile(old_y[None, None, :, None], (bs[0], bs[1], 1, ns[0] * ns[1]))

    prod = torch.einsum(
        "ijkl, ijml -> ijkml", torch.special.sinc(tile_x / delta), torch.special.sinc(tile_y / delta)
    )
    x_tile = torch.tile(x[:, :, :, :, None], (1, 1, 1, 1, ns[0] * ns[1]))

    new_x = torch.einsum("ijklm, ijklm -> ijm", x_tile, prod)
    new_x = new_x.reshape(bs[0], bs[1], ns[0], ns[1])
    return new_x


class GCNN_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        theta=90,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.theta = torch.Tensor([theta])
        self.group_size = int(360 // theta)

        weight = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        torch.nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight)
        A = torch.tensor(
            [
                [torch.cos(self.theta), -torch.sin(self.theta)],
                [torch.sin(self.theta), torch.cos(self.theta)],
            ],
            dtype=torch.float32,
        ).to(self.device)
        self.A = A

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.group_size * out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        device = self.device
        gen_b = torch.empty(
            (
                self.group_size,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            ),
            device=device,
            dtype=torch.float32,
        )

        gen_b[0, :, :, :, :] = self.weight
        for idx in range(1, self.group_size):
            gen_b[idx, :, :, :, :] = sinc_int(
                gen_b[idx - 1, :, :, :, :], self.A, device
            )
        gen_b = gen_b.view(
            self.group_size * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        out = F.conv2d(input, weight=gen_b, stride=self.stride, padding=self.padding)
        return out


# class LGCNN(nn.Module):
#     def __init__(self, in_channels, device):
#         super().__init__()
#         theta_1 = 90
#         self.conv1 = LGCNN_layer(
#             in_channels, 10, kernel_size=3, padding=1, theta=theta_1
#         )

#         group_size_1 = int(360 // theta_1)
#         theta_2 = 90
#         self.conv2 = LGCNN_layer(
#             group_size_1 * 10, 10, kernel_size=3, padding=1, theta=theta_2
#         )

#         group_size_2 = int(360 // theta_2)
#         theta_3 = 90
#         self.conv3 = LGCNN_layer(
#             group_size_2 * 10, 20, kernel_size=3, padding=1, theta=theta_3
#         )

#         group_size_3 = int(360 // theta_3)
#         theta_4 = 90
#         self.conv4 = LGCNN_layer(group_size_3 * 20, 20, kernel_size=3, padding=1)

#         group_size_4 = int(360 // theta_4)
#         # you have to calculate the feature map size for the first fully connected layer
#         self.fc1 = nn.Linear(group_size_4 * 20 * f_map_size * f_map_size, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x, device, training=True):
#         # if you want invariance, you have to pool over the groups
#         x = F.relu(self.conv1(x, device))
#         x = F.relu(self.conv2(x, device))
#         x = F.relu(self.conv3(x, device))
#         x = F.relu(self.conv4(x, device))

#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x)
