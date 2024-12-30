# Bayesian Deep Operator Networks
Bayesian Deep Operator Networks (DeepBayONets) enhance Physics-Informed Neural Networks (PINNs) by incorporating Bayesian methods with Deep Operator Networks (DeepONets) to estimate parameters of an underlying Partial Differential Equation (PDE). This integration allows the networks to deliver more accurate estimations for both forward and inverse problems, even when dealing with noisy data, while also effectively measuring uncertainty in predictions. Additionally, Bayesian Deep Operator Networks substantially decrease the computational workload compared to traditional methods such as Bayesian Neural Networks (BNNs).
<br><br>
<div align="center">
  <img width="512" alt="network architecture graph" src="https://github.com/user-attachments/assets/01f4b821-0fbd-40f6-87da-329dcdea5a2b" style="display: block; margin: 0 auto;"/>
</div>
<br><br>
Three benchmark problems were used to evaluate DeepBayONets:
<ul>
  <li><a href="https://github.com/csml-beach/differentiable-models/blob/3b98dc6ff09f885417caab0b80758b927ec13894/notebooks/func-approximator/func-approx-high-noise.ipynb" target="_blank">One-dimensional function approximation </a></li>
  <li><a href="https://github.com/csml-beach/differentiable-models/blob/3b98dc6ff09f885417caab0b80758b927ec13894/notebooks/heat-equation/bayes-pinn-PDE-posterior-samples.ipynb" target="_blank">One-dimensional unsteady heat equation </a></li>
  <li><a href="https://github.com/csml-beach/differentiable-models/blob/3b98dc6ff09f885417caab0b80758b927ec13894/notebooks/2D-non-linear-diffusion-reaction/2d-nonlinear-diffusion-reaction.ipynb" target="_blank">Two-dimensional reaction-diffusion equation </a></li>
</ul>
<br> <br>
<div align="center">
  <img width="512" src="https://github.com/user-attachments/assets/38881587-441d-4a4e-9ba2-f4dc1d22d4b3" alt="learning process" style="display: block; margin: 0 auto;">
  <img width="512" src="https://github.com/user-attachments/assets/23f327da-2377-407c-a657-b604846be593" alt="parameter distribution learning" style="display: block; margin: 0 auto;">
<div align="center">
  
![samples_github](https://github.com/user-attachments/assets/60cb9064-ff1b-401b-b6ff-50900c817a4a)

