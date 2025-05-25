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
        # self.raw_l = None
        self.lambda_param = None
        # Network

        self.b1 = nn.Linear(3, 200).to(self.device)
        self.b2 = nn.Linear(200, 200).to(self.device)
        self.b3 = nn.Linear(200, 1).to(self.device)

        self.t1 = nn.Linear(3, 200).to(self.device)
        self.t2 = nn.Linear(200, 200).to(self.device)
        self.t3 = nn.Linear(200, 2).to(self.device)


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

    # @property
    # def lambda_param(self):
    #     return torch.exp(self.raw_l)

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
        h = torch.tanh(self.b1(x))
        h = torch.tanh(self.b2(h))
        
        t = torch.tanh(self.t1(x))
        t = torch.tanh(self.t2(t))

        prod = h * t
        u = self.b3(prod).squeeze(-1)
        params = self.t3(prod).squeeze(-1)
        mean = 2+torch.exp(params[:,0]).mean()
        std = torch.exp(params[:,1]).mean()
        self.lambda_param = torch.normal(mean=mean, std=std).to(self.device)

        return u

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
    
    def check_exact_residual(self):
        x = self.interior           # (N,3) with requires_grad=True
        # exact u, keep requires_grad so autograd works
        exact_u = exact_solution(x[:,0], x[:,1], x[:,2])

        lap_exact = self.laplacian(exact_u, x)   # Laplacian of the *exact* field
        lap_exact = lap_exact.squeeze(-1)        # make it (N,) not (N,1)

        k = 4.493409
        residual = -lap_exact - (k**2) * exact_u

        print("max |residual| =", residual.abs().max().item())

    def loss_boundary(self):
        u_b = self.forward(self.boundary)
        return torch.mean(u_b**2)

    def loss_data(self):
        u_d = self.forward(self.data_points)
        return torch.mean((u_d - self.data_values)**2)

    def compute_losses(self):
        return self.loss_interior(), self.loss_boundary(), self.loss_data()
    
# ------------------------------------------------------------
# visualise learned vs. exact on several slices
# ------------------------------------------------------------

def plot_slices(model, device, z0_list=(0.0, 0.3, 0.6)):
    """
    Compare u_pred vs. exact on:
      • axial line (x=y=0)
      • several horizontal slices z = z0
      • meridional slice (x=0 plane)

    z0_list : tuple of z-heights for the equatorial slices
    """
    model.eval()
    with torch.no_grad():
        # ---- 1) axial line (x=y=0) --------------------------
        z = torch.linspace(-1, 1, 300, device=device)
        pts_axis = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], 1)
        u_pred_axis = model(pts_axis).cpu().numpy()
        u_exact_axis = exact_solution(pts_axis[:,0], pts_axis[:,1], pts_axis[:,2]).cpu().numpy()

        plt.figure(figsize=(4,3))
        plt.plot(z.cpu(), u_pred_axis, label='pred')
        plt.plot(z.cpu(), u_exact_axis, '--', label='exact')
        plt.title('Axis (x=y=0)')
        plt.xlabel('z'); plt.ylabel('u')
        plt.legend(); plt.tight_layout()

        # ---- 2) equatorial slices z = z0 --------------------
        for z0 in z0_list:
            # polar grid on the disk r<=sqrt(1-z0²)
            r_max = np.sqrt(max(1e-8, 1 - z0**2))
            theta = torch.linspace(0, 2*np.pi, 361, device=device)
            r = torch.linspace(0, r_max, 200, device=device)
            R, Θ = torch.meshgrid(r, theta, indexing='ij')
            X = R*torch.cos(Θ); Y = R*torch.sin(Θ); Z = torch.full_like(X, z0)
            pts_flat = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], 1)
            u_pred = model(pts_flat).reshape(R.shape).cpu().numpy()
            u_exact = exact_solution(pts_flat[:,0], pts_flat[:,1], pts_flat[:,2]).reshape(R.shape).cpu().numpy()

            fig, axs = plt.subplots(1,2,subplot_kw={'projection':'polar'}, figsize=(6,3))
            for ax, data, ttl in zip(axs, [u_pred, u_exact], ['pred','exact']):
                c=ax.contourf(Θ.cpu(), R.cpu(), data, levels=50)
                ax.set_title(f'{ttl}  z={z0:.2f}')
                fig.colorbar(c, ax=ax, shrink=0.8)
            plt.tight_layout()

        # ---- 3) meridional slice x=0 (y–z plane) ------------
        y = torch.linspace(-1,1,301, device=device)
        z = torch.linspace(-1,1,301, device=device)
        Y, Z = torch.meshgrid(y, z, indexing='ij')
        mask = Y**2 + Z**2 <= 1.0
        X0 = torch.zeros_like(Y)
        pts_plane = torch.stack([X0[mask], Y[mask], Z[mask]], 1)
        u_pred_plane = model(pts_plane).cpu().numpy()
        u_exact_plane = exact_solution(pts_plane[:,0], pts_plane[:,1], pts_plane[:,2]).cpu().numpy()

        # put back into full grid with NaNs outside ball
        U_pred_grid = np.full(Y.shape, np.nan); U_exact_grid = np.full_like(U_pred_grid, np.nan)
        U_pred_grid[mask.cpu().numpy()]  = u_pred_plane
        U_exact_grid[mask.cpu().numpy()] = u_exact_plane

        fig,axs = plt.subplots(1,2,figsize=(6,3))
        for ax, data, ttl in zip(axs,[U_pred_grid,U_exact_grid],['pred','exact']):
            im = ax.imshow(data, origin='lower', extent=[-1,1,-1,1], cmap='seismic')
            ax.set_title(f'{ttl}  slice x=0')
            fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        plt.show()

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
    for epoch in range(10):
        optimizer.zero_grad()
        loss_d = model.loss_data()
        loss_d.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Pretrain epoch {epoch}: data_loss={loss_d.item():.2e}")

    # Full PINN training
    model.w_int  = 0.01
    model.w_bc   = 1.0
    model.w_norm = 10.0
    model.w_data = 10.0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1000):
        optimizer.zero_grad()
        model.create_dataset()
        # model.check_exact_residual()  # check residual before training
        li, lb, ld = model.compute_losses()
        loss = model.w_int*li + model.w_bc*lb + model.w_data*ld
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: PDE={li.item():.2e}, BC={lb.item():.2e}, DATA={ld.item():.2e}, λ={model.lambda_param.mean().item():.4f}")

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

    plot_slices(model, model.device, z0_list=(0.0, 0.25, 0.5))