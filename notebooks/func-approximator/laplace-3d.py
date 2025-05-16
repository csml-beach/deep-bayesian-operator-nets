#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

# Exact solution for ℓ=1,m=0 dipole mode
def exact_solution(x, y, z, k=4.493409, eps=1e-6):
    r = torch.sqrt(x**2 + y**2 + z**2 + eps)
    kr = k * r
    j1 = torch.sin(kr) / (kr**2) - torch.cos(kr) / kr
    return j1 * (z / r)

class Experiment(nn.Module):
    def __init__(self, num_samples=1000, bc_factor=2, num_data=200):
        super().__init__()
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Learnable log-eigenvalue
        self.raw_l = nn.Parameter(torch.log(torch.tensor(20.0, device=self.device)))
        # Network
        self.net = nn.Sequential(
            nn.Linear(3, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200, 1)
        ).to(self.device)
        # Loss weights (set later in curriculum)
        self.w_int = 1.0
        self.w_bc = 1.0
        self.w_data = 1.0
        self.w_norm = 0.1
        # Dataset sizes
        self.num_samples = num_samples
        self.bc_samples = int(num_samples * bc_factor)
        self.num_data = num_data
        # Prepare datasets
        self.create_dataset()

    @property
    def lambda_param(self):
        return torch.exp(self.raw_l)

    def create_dataset(self):
        N = self.num_samples
        R = torch.rand(N,1,device=self.device)**(1/3)
        X = torch.randn(N,3,device=self.device)
        X = X / X.norm(dim=1,keepdim=True)
        self.interior = (X * R).requires_grad_(True)
        M = self.bc_samples
        B = torch.randn(M,3,device=self.device)
        B = B / B.norm(dim=1,keepdim=True)
        self.boundary = B.requires_grad_(True)
        D = self.num_data
        Rd = torch.rand(D,1,device=self.device)**(1/3)
        Xd = torch.randn(D,3,device=self.device)
        Xd = Xd / Xd.norm(dim=1,keepdim=True)
        self.data_points = (Xd * Rd)
        x_d, y_d, z_d = self.data_points[:,0], self.data_points[:,1], self.data_points[:,2]
        self.data_values = exact_solution(x_d, y_d, z_d).detach()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def laplacian(self, u, x):
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        lap = torch.zeros_like(u)
        for i in range(3):
            grad2 = torch.autograd.grad(grad_u[:,i], x, grad_outputs=torch.ones_like(grad_u[:,i]), create_graph=True)[0]
            lap += grad2[:,i]
        return lap

    def loss_interior(self):
        u = self.forward(self.interior)
        lap_u = self.laplacian(u, self.interior)
        res = -lap_u - self.lambda_param * u
        loss_pde = torch.mean(res**2)
        loss_norm = (torch.mean(u**2) - 1.0)**2
        return loss_pde + self.w_norm * loss_norm

    def loss_boundary(self):
        u_b = self.forward(self.boundary)
        return torch.mean(u_b**2)

    def loss_data(self):
        u_d = self.forward(self.data_points)
        return torch.mean((u_d - self.data_values)**2)

    def compute_losses(self):
        return self.loss_interior(), self.loss_boundary(), self.loss_data()

if __name__ == '__main__':
    torch.manual_seed(345)
    # Instantiate
    model = Experiment()
    model.to(model.device)
    print(f"Using device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    # Pretrain on data only
    model.w_int = 0.0
    model.w_bc  = 0.0
    model.w_norm= 0.0
    model.w_data= 1.0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(500):
        optimizer.zero_grad()
        loss_d = model.loss_data()
        loss_d.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Pretrain epoch {epoch}: data_loss={loss_d.item():.2e}")

    # Full PINN training
    model.w_int  = 2.5
    model.w_bc   = 1.0
    model.w_norm = 10.0
    model.w_data = 100.0
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(500):
        optimizer.zero_grad()
        li, lb, ld = model.compute_losses()
        loss = model.w_int*li + model.w_bc*lb + model.w_data*ld
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: PDE={li.item():.2e}, BC={lb.item():.2e}, DATA={ld.item():.2e}, λ={model.lambda_param.item():.4f}")

    # # After your 800-epoch Adam run
    # optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
    # max_iter=1000, tolerance_grad=1e-8, history_size=10)

    # def closure():
    #     optimizer_lbfgs.zero_grad()
    #     li, lb, ld = model.compute_losses()
    #     loss = model.w_int*li + model.w_bc*lb + model.w_data*ld
    #     loss.backward()
    #     return loss


    # print("Starting L-BFGS fine-tuning...")
    # optimizer_lbfgs.step(closure)
    # li, lb, ld = model.compute_losses()
    # print(f"After L-BFGS: PDE={li.item():.2e}, BC={lb.item():.2e}, DATA={ld.item():.2e}, λ={model.lambda_param.item():.4f}")

    # Visualize
    z = torch.linspace(-1,1,200,device=model.device)
    pts = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], dim=1)
    with torch.no_grad():
        u_pred = model(pts).cpu().numpy()
    z_np = z.cpu().numpy()
    u_ex = exact_solution(pts[:,0],pts[:,1],pts[:,2]).cpu().numpy()
    plt.plot(z_np, u_pred, label='Predicted')
    plt.plot(z_np, u_ex, '--', label='Exact')
    plt.legend(); plt.show()