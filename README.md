# Bayesian Deep Operator Networks
Bayesian Deep Operator Networks (DeepBayONets) enhance Physics-Informed Neural Networks (PINNs) by incorporating Bayesian methods with Deep Operator Networks (DeepONets) to estimate parameters of an underlying Partial Differential Equation (PDE). This integration allows the networks to deliver more accurate estimations for both forward and inverse problems, even when dealing with noisy data, while also effectively measuring uncertainty in predictions. Additionally, Bayesian Deep Operator Networks substantially decrease the computational workload compared to traditional methods such as Bayesian Neural Networks (BNNs).
<br><br>
<img src="https://github.com/user-attachments/assets/617bcf7f-2f3f-4fa1-9238-1f5b3f4c66ea" alt="architecture" />
<br><br>
Three benchmark problems were used to evaluate DeepBayONets:
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/func-approximator/func-approx-high-noise.ipynb" target="_blank">One-dimensional function approximation </a>
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/notebooks/heat-equation/bayes-pinn-PDE-posterior-samples.ipynb" target="_blank">One-dimensional unsteady heat equation </a>
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/notebooks/2D-non-linear-diffusion-reaction/2d-non-linear-multimode.ipynb" target="_blank">Two-dimensional reaction-diffusion equation </a> <br> <br> 
![samples_github](https://github.com/user-attachments/assets/60cb9064-ff1b-401b-b6ff-50900c817a4a)

