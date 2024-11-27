# Bayesian Deep Operator Networks
Bayesian Deep Operator Networks (DeepBayONets) enhance Physics-Informed Neural Networks (PINNs) by incorporating Bayesian methods with Deep Operator Networks (DeepONets) to estimate parameters of an underlying Partial Differential Equation (PDE). This integration allows the networks to deliver more accurate parameter estimations for both forward and inverse problems, even when dealing with noisy data, while also effectively measuring uncertainty in predictions. Additionally, Bayesian Deep Operator Networks substantially decrease the computational workload compared to traditional methods such as Bayesian Neural Networks (BNNs).
<br><br>
To evaluate DeepBayONets, three benchmark problems were utilized:
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/func-approximator/func-approx-high-noise.ipynb" target="_blank">One-dimensional function approximation </a>
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/notebooks/heat-equation/bayes-pinn-PDE-posterior-samples.ipynb" target="_blank">One-dimensional unsteady heat equation </a>
- <a href="https://github.com/csml-beach/differentiable-models/blob/main/notebooks/2D-non-linear-diffusion-reaction/2d-non-linear-multimode.ipynb" target="_blank">Two-dimensional reaction-diffusion equation </a>

![image002](https://github.com/csml-beach/differentiable-models/assets/5168326/6b0c0fcd-3353-4eee-9b1b-1961d88f132a)
![image003](https://github.com/csml-beach/differentiable-models/assets/5168326/13bb14b0-268e-4ae6-bb77-f062ecd97a75)
![image004](https://github.com/csml-beach/differentiable-models/assets/5168326/2106bd8b-5695-4c9f-a5dc-d980bff2074f)
<img width="682" alt="PDE" src="https://github.com/csml-beach/differentiable-models/assets/5168326/93ce91e5-1719-472f-aac0-a756d1967d1c">
![architecture](https://github.com/user-attachments/assets/617bcf7f-2f3f-4fa1-9238-1f5b3f4c66ea)
