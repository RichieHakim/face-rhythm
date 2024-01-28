import torch
import torch.nn as nn

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        return x[0][0], x[0][1]


## Write a Lucas Kanade class
class LucasKanade(nn.Module):
    def __init__(
        self,
        patch_size=15,
        num_iters=10,
        num_samples=100,
        learning_rate=0.1,
        device="cuda",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.device = device

        self.sobel = Sobel().to(self.device)
        self.warp = nn.functional.grid_sample

    def forward(self, img1, img2, points1):
        """
        img1: (1, 1, H, W)
        img2: (1, 1, H, W)
        points1: (N, 2)
        """
        ## Move points to device
        points1 = points1.to(self.device)
        ## Move images to device
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        ## Get gradients
        # .split(1, dim=1)
        Ix, Iy = self.sobel(img1[:, :])
        Ix = Ix.squeeze(1)
        Iy = Iy.squeeze(1)

        ## Get patches
        patches1 = self.get_patches(img1, points1)
        patches2 = self.get_patches(img2, points1)

        ## Compute Hessian
        H = self.compute_hessian(patches1, Ix, Iy)

        ## Compute steepest descent
        sd = self.compute_steepest_descent(patches1, Ix, Iy)

        ## Compute error
        error = self.compute_error(patches1, patches2)

        ## Compute delta_p
        delta_p = self.compute_delta_p(H, sd, error)

        return delta_p

    def get_patches(self, img, points):
        """
        img: (1, 1, H, W)
        points: (N, 2)
        """
        ## Get patches
        patches = []
        for point in points:
            x, y = point
            x = int(x)
            y = int(y)
            patch = img[:, :, y : y + self.patch_size, x : x + self.patch_size]
            patches.append(patch)
        patches = torch.cat(patches, dim=0)
        return patches

    def compute_hessian(self, patches, Ix, Iy):
        """
        patches: (N, 1, patch_size, patch_size)
        Ix: (1, H, W)
        Iy: (1, H, W)
        """
        N = patches.shape[0]
        H = []
        for i in range(N):
            patch = patches[i]
            Ix_patch = self.get_patches(Ix, patch)
            Iy_patch = self.get_patches(Iy, patch)
            Ix_patch = Ix_patch.view(1, -1)
            Iy_patch = Iy_patch.view(1, -1)

            h = torch.cat([Ix_patch, Iy_patch], dim=0)
            H.append(h)
        H = torch.stack(H, dim=0)
        return H

    def compute_steepest_descent(self, patches, Ix, Iy):
        """
        patches: (N, 1, patch_size, patch_size)
        Ix: (1, H, W)
        Iy: (1, H, W)
        """
        N = patches.shape[0]
        sd = []
        for i in range(N):
            patch = patches[i]
            Ix_patch = self.get_patches(Ix, patch)
            Iy_patch = self.get_patches(Iy, patch)
            Ix_patch = Ix_patch.view(1, -1)
            Iy_patch = Iy_patch.view(1, -1)

            sd_i = torch.cat([Ix_patch, Iy_patch], dim=0)
            sd.append(sd_i)
        sd = torch.stack(sd, dim=0)
        return sd

    def compute_error(self, patches1, patches2):
        """
        patches1: (N, 1, patch_size, patch_size)
        patches2: (N, 1, patch_size, patch_size)
        """
        N = patches1.shape[0]
        error = []
        for i in range(N):
            patch1 = patches1[i]
            patch2 = patches2[i]
            error_i = patch2 - patch1
            error_i = error_i.view(1, -1)
            error.append(error_i)
        error = torch.stack(error, dim=0)
        return error

    def compute_delta_p(self, H, sd, error):
        """
        H: (N, 2, patch_size * patch_size)
        sd: (N, 2, patch_size * patch_size)
        error: (N, 1, patch_size * patch_size)
        """
        N = H.shape[0]
        delta_p = []
        for i in range(N):
            h = H[i]
            sd_i = sd[i]
            error_i = error[i]

            h_inv = torch.inverse(h)
            sd_i = sd_i.permute(1, 0)
            delta_p_i = h_inv @ sd_i @ error_i
            delta_p_i = delta_p_i.permute(1, 0)
            delta_p.append(delta_p_i)
        delta_p = torch.stack(delta_p, dim=0)
        return delta_p