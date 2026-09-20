# Posterior-Conditioned Latent Diffusion for Discrete Transmission and Continuous Reconstruction in Generative Semantic Communication


## Abstract
Generative semantic communication provides a promising paradigm for semantic transmission by leveraging generative models. Existing generative semantic communication frameworks either adopt continuous latent representation prone to channel quantization distortion, or discrete token transmission suffering from reconstruction quality degradation.

To tackle this issue, we propose a posterior-conditioned latent diffusion framework for generative semantic communication, which supports discrete transmission and continuous reconstruction. The transmitter extracts posterior latent features and quantizes them into discrete codewords for transmission over wireless channels. At the receiver, a latent diffusion model conditioned on the received discrete codewords reconstructs continuous semantic representations.

Our framework combines the error robustness of discrete transmission and the high-fidelity reconstruction capability of continuous generative models. Experimental results demonstrate that our proposed framework achieves superior semantic reconstruction performance under various channel conditions compared with baseline semantic communication schemes.




<p align="center">
  <img src="./figs/fig_model.pdf" width="800">
</p>
