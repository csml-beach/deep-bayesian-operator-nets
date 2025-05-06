import torch

# Pick a few random points in ℝ³
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 50
X = torch.randn(N, 3, device=device, requires_grad=True)

# Define the test function u = x^2 + y^2 + z^2
u = (X**2).sum(dim=1)            # shape (N,)

# Compute the gradient ∇u (shape (N,3))
grad_u = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u),
                             create_graph=True)[0]

# Compute the Laplacian = ∑₁³ ∂ᵢ²u by differentiating each component again
lap = torch.zeros_like(u)
for i in range(3):
    grad2 = torch.autograd.grad(grad_u[:, i], X,
                                grad_outputs=torch.ones_like(grad_u[:, i]),
                                create_graph=True)[0]
    lap += grad2[:, i]

# The analytic Laplacian is constant 6
lap_analytic = torch.full_like(lap, 6.0)

# Check maximum and mean absolute error
err = (lap - lap_analytic).abs()
print(f"Max error:  {err.max().item():.2e}")
print(f"Mean error: {err.mean().item():.2e}")