I am machine learning scientist working at Alexa, Amazon. 
I received my PhD in statistical machine learning from the University of Alberta, Canada, under supervision of Prof. Dale Shuurmans.
I worked on diverse fields of AI/ML including deep generative models on different data modalities such as image and graphs.

# Background and Research Interests

- Generative Models 
- Graph Neural Networks   
- NLP and Language Models
- Computer Vision
- Multi-Modal/Multi-View Learning 
- Federated Learning
- Reinforcement Learning
- Bayesian Optimization

<!-- | Generative Models    | NLP and Sequence Modeling |
| -------- | ------- |
| Graph Neural Networks | Representation Learning  |
| Probabilistic Deep Learning | Federated Learning |
| Computer Vision | Reinforcement Learning|
| Multi-Modal/Multi-View Learning | Bayesian Optimization| -->

# Publications and Research
### Graph Generative Networks
<details>
<summary> Abstract </summary>
  Most real-world graphs exhibit a hierarchical structure, which is often overlooked
by existing graph generation methods. To address this limitation, we propose a
novel graph generative network that captures the hierarchical nature of graphs
and successively generates the graph sub-structures in a coarse-to-fine fashion. At
each level of hierarchy, this model generates communities in parallel, followed by
the prediction of cross-edges between communities using a separate model. This
modular approach results in a highly scalable graph generative network. 
  
  Moreover, we model the output distribution of edges in the hierarchical graph with
a multinomial distribution and derive a recursive factorization for this distribution,
enabling us to generate sub-graphs with integer-valued edge weights in an
autoregressive approach. Empirical studies demonstrate that the proposed generative
model can effectively capture both local and global properties of graphs
and achieves state-of-the-art performance in terms of graph quality on various
benchmarks.
</details>

- **M. Karami**, “HiGen: Hierarchical Graph Generative Networks”, *arXiv preprint arxiv:2305.19337,* (2023).
[paper](https://arxiv.org/abs/2305.19337) [code](https://github.com/Karami-m/HiGen_main)
- **M. Karami**, J. Luo, “On Hierarchical Multi-Resolution Graph Generative Models ”, *arXiv preprint arxiv:2303.03293,
Machine Learning for Drug Discovery (MLDD) Workshop ICLR 2023* (2023).

### Generative Models
<details>
<summary> Abstract </summary>
  Normalizing flows construct a complex probability density by transforming a simple base density, such as a standard normal distribution, via a chain of smooth, invertible mappings (bijections). Flow-based generative networks can be used to construct high quality generative probabilistic models, but training and sample generation require repeated evaluation of Jacobian determinants and function inverses. In this work, we investigated a set of novel normalizing flows based on circular and symmetric convolutions. It was shown that these transforms admit efficient Jacobian determinant computation and inverse mapping (deconvolution) in 𝒪(𝑁 log𝑁) time. Based on these invertible convolution filters, a nonlinear data-adaptive convolution transformation was proposed where expressiveness is increased by allowing a layer’s kernel to adapt to the layers input.

Another outcome of this work was an analytic approach to designing and also better understanding the role of nonlinear gates through the lens of their contribution to latent variables’ distributions. We have shown that specific regularizers, such as sparsity, can be induced on intermediate activations by designing customized pointwise nonlinear gates.
</details>

- **M. Karami**, D. Schuurmans, J. Sohl-Dickstein, D. Duckworth, L. Dinh, “Invertible Convolutional Flow”, *Advances in Neural Information Processing Systems (NeurIPS) 2019*, Vancouver, Canada **(spotlight presentation top 3.5%)** (2019).
[paper](https://papers.nips.cc/paper/2019/hash/b1f62fa99de9f27a048344d55c5ef7a6-Abstract.html)
- **M. Karami**, D. Schuurmans, J. Sohl-Dickstein, D. Duckworth, L. Dinh, “Symmetric Convolutional Flow”,
*Workshop on Invertible Neural Nets and Normalizing Flows (INNF), ICML 2019* (2019).
- **M. Karami**, L. Dinh, D. Duckworth, J. Sohl-Dickstein, D. Schuurmans, “Generative Convolutional Flow for
Density Estimation”, *Workshop on Bayesian Deep Learning, NeurIPS 2018*, Montreal, Canada (2018).

### Federated Learning and Privacy Preserving ML
- H. Yu, K. Guo, M. Karami, X. Chen, G. Zhang, P. Poupart, “Federated Bayesian Neural Regression: A
Scalable Global Federated Gaussian Process”, *arXiv preprint arxiv:2206.06357,* (2022).
- M. Hassan, Z. Zhang, k. Guo, M. Karami, G. Zhang, X. Chen, P. Poupart, “Robust One Round Federated
Learning with Predictive Space Bayesian Inference”, *arXiv preprint arxiv:2206.09526,* (2022).
- D. Jiang, G. Zhang, M. Karami, X. Chen, Y. Shao, Y. Yu, “DP2-VAE: Differentially Private Pre-trained
Variational Autoencoders”, *arXiv preprint arxiv:2208.03409,* (2022).

### Deep Generative Multi-View (Multi-Modal) Learning
<details>
<summary> Abstract </summary>
  We proposed an interpretable deep generative framework for multi-view learning based on a probabilistic formulation of canonical correlation analysis (CCA). The model combines a linear multi-view layer in the latent space with deep generative networks as observation models. The proposed model decomposes the variability between views into a shared latent representation that describes the common underlying sources of variation and a set of view-specific components. We designed an efficient learning algorithm using a variational inference procedure incorporating the solution of probabilistic CCA. This also offered a flexible data fusion method in the latent space. Importantly, the proposed model can be generalized to an arbitrary number of views. An empirical analysis confirms that the proposed deep multi-view model can discover subtle relationships between multiple views and recover rich representations.
</details>

- **M. Karami**, D. Schuurmans, “Deep Probabilistic Canonical Correlation Analysis”, *AAAI
conf. on Artificial Intelligence 2021*, available on *arXiv preprint arXiv:2003.04292*, (2021).
[paper](https://papers.nips.cc/paper/2017/hash/c2964caac096f26db222cb325aa267cb-Abstract.html)
[code](https://github.com/Karami-m/Deep-Probabilistic-Multi-View)
- **M. Karami**, “Deep Generative Multi-view Learning”, Accepted for *the workshop Data and Machine Learning
Advances with Multiple Views, ECML/PKDD 2019*, Wurzburg, Germany (2019).

### Time Series Analysis
<details>
<summary> Abstract </summary>
  Maximum likelihood is typically considered to be hard in this setting since latent states and transition parameters must be inferred jointly. Given that expectation-maximization does not scale and is prone to local minima, moment-matching approaches from the subspace identification literature have become standard, despite known statistical efficiency issues. In this work, we instead reconsidered likelihood maximization of LDS with generalized-linear observation models. Key to the approach was a reformulation of the LDS model as a two-view convex optimization problem that allowed us to approximate the estimation task as a form of matrix factorization, and hence apply recent global optimization techniques. Furthermore, a novel proximal mapping update was analytically derived for this two-view reformulation that significantly simplified the optimization procedure. The resulting algorithm was simple to use and flexible enough to incorporate different losses and regularizers while empirical studies demonstrated that this estimation strategy outperforms widely-used identification algorithms such as subspace identification methods, both in terms of accuracy and runtime.
</details>

- **M. Karami**, M. White, D. Schuurmans, C. Szepesvari, “Multi-view Matrix Factorization for Linear Dynamical
System Estimation”, *Advances in Neural Information Processing Systems (NIPS)*, (2017).
[paper](https://papers.nips.cc/paper/2017/hash/c2964caac096f26db222cb325aa267cb-Abstract.html)
- **M. Karami**, M. White, D. Schuurmans, “Optimal Linear Dynamical System Identification”, *NIPS Time
Series Workshop, Barcelona, Spain, Dec. 2016* **Best poster award** (2016).

### Wireless Communications
- **M. Karami**, A. Olfat, N. C. Beaulieu, “Pilot Symbol Parameter Optimization Based on Imperfect Channel
State Prediction for OFDM Systems”, *IEEE Transactions on Communications, vol.61, no.6, pp.2557,2567*,
June (2013).
- **M. Karami** and N. C. Beaulieu, “Channel adaptive power allocation and pilot optimization for OFDM
systems”, *IEEE Global Comm. Conf. (GLOBECOM)*, Anaheim, CA, Dec. (2012).
- **M. Karami**, A. Olfat and N. C. Beaulieu, “Pilot Symbol Assisted Adaptive Modulation for OFDM Systems
with Imperfect Channel State Information”, in Proc. *IEEE Global Comm. Conf. (GLOBECOM)*, (2010).
- **M. Karami**, A. Olfat, “Fast Blind Adaptive Channel Shortening Using Signal Subspace”, *IEEE VTC 2008
Spring*, Singapore, pp. 2621-2625, May (2008).
- **M. Karami**, A. Olfat, “Fast Subspace-based Adaptive Channel Shortening for Multicarrier Systems”, *16th
Iranian Conference on Electrical Engineering (ICEE)*, Tehran, Iran, May (2008).

# Educations
- **Ph.D. in Statistical Machine Learning**, Computing Science, University of Alberta, 2014-2020
  - Supervisor: Prof. D. Schuurmans
  - Thesis: **Advances in Probabilistic Generative Models**

- **M.Sc. Electrical and Computer Engineering**, University of Tehran, 2005-2008
  - Supervisor: Dr. A. Olfat (aolfat@ut.ac.ir)
  - Thesis: Adaptive Pilot Symbol Assisted Modulation (PSAM) for OFDM Systems

- **B.Sc. Electrical and Computer Engineering**, Isfahan University of Technology, 2001-2005
