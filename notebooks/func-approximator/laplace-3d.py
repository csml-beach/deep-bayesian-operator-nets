import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
from sys import stderr
from scipy import stats
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.interpolate import griddata
from matplotlib.pyplot import scatter, figure
from torch.utils.data import TensorDataset, DataLoader
from scipy.special import spherical_jn

# Set random seeds for reproducibility
debug = False
seed = 345
np.random.seed(seed)
torch.manual_seed(seed)

# Exact solution function for m=0
def exact_solution(x, y, z, k):
    # Move tensors to CPU before computation
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    z_cpu = z.cpu()
    
    r = torch.sqrt(x_cpu**2 + y_cpu**2 + z_cpu**2)
    theta = torch.acos(z_cpu/r)
    
    # Convert to numpy for scipy function
    r_np = r.numpy()
    jn = spherical_jn(1, k*r_np).astype(np.float32)
    
    # Convert back to tensor and move to original device
    jn_tensor = torch.from_numpy(jn).to(x.device)
    return jn_tensor * torch.cos(theta).to(x.device)

# Set up matplotlib and seaborn
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(4, 4)})
sns.set_style("whitegrid")

class Experiment(nn.Module):
    def __init__(self):
        super(Experiment, self).__init__()
        
        # Set device (GPU if available, otherwise MPS for Apple Silicon, else CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model parameters
        self.optimizer = None
        self.train_loss_history = []
        self.w_int = 1
        self.w_data = 1
        self.w_bc = 1
        self.w_param_std = 1
        self.numInputs = 3  # x, y, z coordinates
        self.numOutputs = 1
        self.hidden_size = 200
        self.numParams = 1

        # Initialize history variables
        self.total_loss_history = []
        self.loss_interior_history = []
        self.loss_data_history = []
        self.loss_bc_history = []
        self.loss_std_history = []
        self.loss_boundary_history = []

        # Initialize eigenvalue as a learnable parameter
        self.predicted_params = None
        self.mean_predicted_params = None
        self.mu = torch.tensor([3.0], device=self.device).requires_grad_(True)
        self.var = torch.tensor([1.0], device=self.device).requires_grad_(True)

        # Define neural network layers
        self.b1 = nn.Linear(self.numInputs, self.hidden_size).to(self.device)
        self.b2 = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.b3 = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.b4 = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.out = nn.Linear(self.hidden_size, 2*self.numOutputs).to(self.device)
        
        self.act = nn.Tanh()

        

    def create_dataset(self, num_samples=100):
        # Generate points inside the unit sphere
        points = torch.randn((num_samples, 3), device=self.device)
        points = points / torch.norm(points, dim=1, keepdim=True) * torch.rand((num_samples, 1), device=self.device)
        
        # Generate points on the surface
        surface_points = torch.randn((num_samples//4, 3), device=self.device)
        surface_points = surface_points / torch.norm(surface_points, dim=1, keepdim=True)
        
        self.interior_points = points.requires_grad_(True)
        self.surface_points = surface_points.requires_grad_(True)

    def forward(self, x):
        # x is a tensor of shape (batch_size, 3) containing (x,y,z) coordinates
        h1 = self.act(self.b1(x))
        h2 = self.act(self.b2(h1))
        h3 = self.act(self.b3(h2))
        h4 = self.act(self.b4(h3))
        u = self.out(h4)
        return u[:,0], u[:,1]  # mean and log variance
    
    def sample_posterior(self, x):
        # eps = torch.rand(x.shape[0], self.hidden_size, device=self.device).requires_grad_(True)

        t1 = self.act(self.t1(x))
        t2 = self.act(self.t2(t1))


        t3 = self.act(self.t3(t2))
        t4 = self.t4(t3)

        predicted_params = torch.exp(t4)
        posterior_samples = predicted_params


        # posterior_samples = mu.view(-1,1) + torch.multiply(eps,var.view(-1,1))
        #posterior_samples = self.mu + prior.mean(dim=1)*self.var
        # print('shape of posterior_samples:', posterior_samples.shape)
        
        self.update_predicted_params(posterior_samples)
        return t2
    
    def update_predicted_params(self, posterior_samples):
        self.predicted_params = posterior_samples.mean(dim=1)
        mean = torch.mean(posterior_samples)  # Compute the mean along the first axis
        std = torch.std(posterior_samples)    # Compute the standard deviation along the first axis
        self.mean_predicted_params = mean  # Store the mean
        self.std_params = std  # Attach the standard deviation as an attribute
    
    def laplacian(self, u, x):
        # Compute Laplacian using automatic differentiation
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        laplacian_u = 0.0
        for i in range(3):
            grad_u_i = grad_u[:, i:i+1]
            grad2_u_i = torch.autograd.grad(grad_u_i, x, grad_outputs=torch.ones_like(grad_u_i),
                                          create_graph=True, retain_graph=True)[0]
            laplacian_u += grad2_u_i[:, i:i+1]
        return laplacian_u

    def PDE_residual(self, x):
        u, log_var = self.forward(x)
        laplacian_u = self.laplacian(u, x)
        residual = laplacian_u + 20.19 * u 
        return residual, log_var, u

    def loss_interior(self, num_samples=5000):
        self.create_dataset(num_samples) 
        res1, log_var, u = self.PDE_residual(self.interior_points)
        loss_residual1 = torch.mean(res1**2) + 0.1*torch.mean(torch.abs(torch.log(u**2)))
        # loss_residual1 = ngll(res1, torch.zeros_like(res1), torch.exp(log_var))
        return loss_residual1

    def loss_boundary(self):
        u, _ = self.forward(self.surface_points)
        loss_bc = torch.mean(u**2)
        return loss_bc

    def compute_losses(self):
        loss_interior = self.loss_interior()
        loss_boundary = self.loss_boundary()
        return loss_interior, loss_boundary

    def closure(self):
        self.optimizer.zero_grad()
        loss_interior, loss_boundary = self.compute_losses()
        total_loss = self.w_int * loss_interior + self.w_bc * loss_boundary
        total_loss.backward(retain_graph=True)
        return total_loss

    def train(self, epochs, optimizer='Adam', **kwargs):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), **kwargs)
        elif optimizer == 'L-BFGS':
            self.optimizer = torch.optim.LBFGS(self.parameters(), **kwargs)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
            self.optimizer.step(self.closure)
            if epoch % 200 == 0:
                loss_interior, loss_boundary = self.compute_losses()
                total_loss = loss_interior + loss_boundary
                scheduler.step(total_loss)

                self.total_loss_history.append(total_loss.item())
                self.loss_interior_history.append(loss_interior.item())
                self.loss_boundary_history.append(loss_boundary.item())

                print(f'Epoch({optimizer}):{epoch}, Total Loss:{total_loss.item():g} '
                      f'PDE Loss:{loss_interior.item():.4f} '
                      f'BC Loss: {loss_boundary.item():.4f} '
                      f'Lambda: {20.19:.4f}')

def make_plot(model, device):
    # Generate points along the z-axis for visualization
    z = torch.linspace(-1, 1, 100, device=device, dtype=torch.float32)
    x = torch.zeros_like(z)
    y = torch.zeros_like(z)
    points = torch.stack([x, y, z], dim=1)
    
    model.eval()
    with torch.no_grad():
        u_pred, _ = model.forward(points)
    
    # Move tensors to CPU before numpy conversion
    z_np = z.cpu().numpy()
    u_pred_np = u_pred.cpu().numpy()
    model.mean_predicted_params = torch.tensor([20.19], device=device)
    # Compute exact solution
    k = np.sqrt(model.mean_predicted_params.item())
    u_exact = exact_solution(x, y, z, k)
    u_exact_np = u_exact.cpu().numpy()
    
    fig, ax = plt.subplots()
    ax.plot(z_np, u_pred_np, label='Predicted')
    ax.plot(z_np, u_exact_np, label='Exact')
    
    plt.tight_layout()
    plt.legend()
    plt.show()
    return ax

if __name__ == "__main__":
    # Initialize the model
    ngll = torch.nn.GaussianNLLLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    
    print(f"Using device: {device}")
    
    net = Experiment()
    net.to(net.device)
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
    
    # Create dataset
    net.create_dataset(num_samples=5000)
    
    # Set loss weights
    net.w_int = 1.0
    net.w_bc = 1.0
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Train the model
    net.train(500, optimizer='Adam', lr=1e-2)
    
    # Make final plot
    make_plot(net, device) 