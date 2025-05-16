import torch

# Hard-coded root and eigenvalue
k = 4.493409
lambda_exact = k**2

# Generate a batch of random interior points in the unit ball
N = 200
X = torch.randn(N, 3, requires_grad=True)
R = torch.rand(N, 1)**(1/3)
X = X / X.norm(dim=1, keepdim=True) * R
X = X.to(torch.device("cpu")).requires_grad_(True)

# Split coords
x, y, z = X[:,0], X[:,1], X[:,2]

# Define exact solution torch-compatible
def exact_solution_torch(x, y, z, k, eps=1e-8):
    r = torch.sqrt(x**2 + y**2 + z**2 + eps)
    kr = k * r
    j1 = torch.sin(kr)/(kr**2) - torch.cos(kr)/kr
    return j1 * (z/r)

# Compute u and its Laplacian
u_exact = exact_solution_torch(x, y, z, k)
# reuse your laplacian routine (for example, model.laplacian or inline here)
def laplacian(u, X):
    grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    lap = torch.zeros_like(u)
    for i in range(3):
        grad2 = torch.autograd.grad(grad_u[:, i], X,
                                     grad_outputs=torch.ones_like(grad_u[:, i]),
                                     create_graph=True)[0]
        lap += grad2[:, i]
    return lap

lap_u = laplacian(u_exact, X)

# Form the residual
residual = -lap_u - lambda_exact * u_exact

# Inspect maximum absolute residual
print("max |residual| =", residual.abs().max().item())