I am a **Machine Learning/AI Scientist** holding a PhD in **Statistical Machine Learning** with over 14 years of
experience in research and industry. I have a strong background in various AI/ML domains, such as **Deep
Generative Models, NLP, Computer Vision, Efficient AI, and Graph Neural Networks**. My works were published in top
AI conferences in collaboration with leading researchers from Google Brain and DeepMind. I also have valuable
industrial experience in deep learning, NLP, Foundation Models with proficiency in the Python, particularly
PyTorch and TensorFlow. I possess a unique combination of problem-solving, creativity, out-of-the-box
thinking, passion for learning, project management and strong communication skills.

# Work and Professional Experience
- **Researcher**, University of Waterloo (_July 2023 - Present_)
  - Large Language models
  - Efficient and Scalable Systems for ML
  - Graph Generative Networks (GGN)
- **Applied Scientist II**, Alexa, Amazon (_Sep. 2022 - July 2023_)
  - Research and Implementation of efficient pre-trained language models (LLM) for natural language Understanding
(NLU) and named-entity recognition (NER).
- **Senior Machine Learning Researcher**, Noah’s Ark Lab, Huawei (_Feb. 2021- Sep. 2022_)
  - Graph Neural Networks (GNN)
  - Federated and Privacy Preserving Machine Learning
  - Leading ML-based Logic Synthesis Circuits Optimization and Compiler Optimization
- **Guest Researcher**, AMLAB, University of Amsterdam (_Apr. 2018- Apr. 2019_)
  - Research on Deep Generative Networks and Invertible Convolutional Flows
- **Research assistant**, Reinforcement Learning & AI (RLAI) Lab, CS, UofA (_Jan. 2014 - Aug. 2020_)
  - Deep Generative Models, Convolutional Normalizing Flows
  - Deep Generative Multi-Modal/Multi-View Learning,
  - Linear Dynamical Systems, Time Series, Convex Optimization
- Quantum Machine Learning Program, Rotman School of Management, U of Toronto (_Sep. 2017_)
- Deep Learning Summer School, Universite de Montreal Jun. 2017
- Lab instructor (LI) and Teaching assistant (TA), University of Alberta (_Jan. 2011 - Dec. 2016_)
  - Advanced Digital Logic Design (FPGA), Computer Programming
- Research assistant, iCORE Wireless Com. Lab, ECE, UofA (_Jan. 2010 - Dec. 2013_)
  - Adaptive resource allocation in Wireless Systems, Wireless Cognitive Radio

# Background and Research Interests

- Generative Models
- NLP and Large Language Models
- Efficient AI
- Graph Neural Networks
- AI for Science
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

### Large Language Models & Efficient Systems for ML

- **M. Karami**, A. Ghodsi, “Orchid: Flexible and Data-Dependent Convolution for Sequence Modeling
”, [arxiv:2402.18508](https://arxiv.org/abs/2402.18508) _ICLR 2024
Workshop on Mathematical and Empirical Understanding of Foundation Models_, (2024). 
<details>
<summary> Abstract </summary>
 In the rapidly evolving landscape of deep learning, the quest for models that
balance expressivity with computational efficiency has never been more
critical. This paper introduces Orchid, a novel architecture that reimagines
sequence modeling by incorporating a new data-dependent convolution mechanism.
Orchid is designed to address the inherent limitations of traditional attention
mechanisms, particularly their quadratic complexity, without compromising the
ability to capture long-range dependencies and in-context learning. At the core
of Orchid lies the data-dependent convolution layer, which dynamically adjusts
its kernel conditioned on input data using a dedicated conditioning neural
network. We design two simple conditioning networks that maintain shift
equivariance in the adaptive convolution operation. The dynamic nature of
data-dependent convolution kernel, coupled with gating operations, grants
Orchid high expressivity while maintaining efficiency and quasilinear
scalability for long sequences. We rigorously evaluate Orchid across multiple
domains, including language modeling and image classification, to showcase its
performance and generality. Our experiments demonstrate that Orchid
architecture not only outperforms traditional attention-based architectures
such as BERT and Vision Transformers with smaller model sizes, but also extends
the feasible sequence length beyond the limitations of the dense attention
layers. This achievement represents a significant step towards more efficient
and scalable deep learning models for sequence modeling.
</details>

- **M. Karami**, A. Behrouz, “Enhancing Sequence Modeling with Multi-Resolution State Space Models”, _ICML 2024
Workshop on Next Generation of Sequence Modeling Architectures, (2024)_.

### Graph Generative Networks
<details>
<summary> Abstract </summary>
  Most real-world graphs exhibit a hierarchical structure, which is often overlooked by existing graph generation methods. To address this limitation, we propose a novel graph generative network that captures the hierarchical nature of graphs and successively generates the graph sub-structures in a coarse-to-fine fashion. At each level of hierarchy, this model generates communities in parallel, followed by the prediction of cross-edges between communities using separate neural networks. This modular approach enables scalable graph generation for large and complex graphs. 

  
  Moreover, we model the output distribution of edges in the hierarchical graph with a multinomial distribution and derive a recursive factorization for this distribution. This enables us to generate community graphs with integer-valued edge weights in an autoregressive manner. Empirical studies demonstrate the effectiveness and scalability of our proposed generative model, achieving state-of-the-art performance in terms of graph quality across various benchmark datasets.

<!--  Most real-world graphs exhibit a hierarchical structure, which is often overlooked
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
benchmarks. -->

</details>

- **M. Karami**, “HiGen: Hierarchical Graph Generative Networks”, *International
Conference on Learning Representations (**ICLR**),* (2024).
[paper](https://arxiv.org/abs/2305.19337) [code](https://github.com/Karami-m/HiGen_main)
<!--  - **M. Karami**, J. Luo, “On Hierarchical Multi-Resolution Graph Generative Models ”, *arXiv preprint arxiv:2303.03293,
Machine Learning for Drug Discovery (MLDD) Workshop ICLR 2023* (2023). -->

- **M. Karami**, I. Krawczuk, V. Cevher, “Multi-Resolution Graph Diffusion”, _ICLR 2024 Workshop on Machine
Learning for Genomics Explorations_, (2024).

### Generative Models
<details>
<summary> Abstract </summary>
  Normalizing flows construct a complex probability density by transforming a simple base density, such as a standard normal distribution, via a chain of smooth, invertible mappings (bijections). Flow-based generative networks can be used to construct high quality generative probabilistic models, but training and sample generation require repeated evaluation of Jacobian determinants and function inverses. In this work, we investigated a set of novel normalizing flows based on circular and symmetric convolutions. It was shown that these transforms admit efficient Jacobian determinant computation and inverse mapping (deconvolution) in 𝒪(𝑁 log𝑁) time. Based on these invertible convolution filters, a nonlinear data-adaptive convolution transformation was proposed where expressiveness is increased by allowing a layer’s kernel to adapt to the layers input.

Another outcome of this work was an analytic approach to designing and also better understanding the role of nonlinear gates through the lens of their contribution to latent variables’ distributions. We have shown that specific regularizers, such as sparsity, can be induced on intermediate activations by designing customized pointwise nonlinear gates.
</details>

- **M. Karami**, D. Schuurmans, J. Sohl-Dickstein, D. Duckworth, L. Dinh, “Invertible Convolutional Flow”, *Advances in Neural Information Processing Systems (**NeurIPS**) 2019*, Vancouver, Canada **(spotlight presentation top 3.5%)** (2019).
[paper](https://papers.nips.cc/paper/2019/hash/b1f62fa99de9f27a048344d55c5ef7a6-Abstract.html)
- **M. Karami**, D. Schuurmans, J. Sohl-Dickstein, D. Duckworth, L. Dinh, “Symmetric Convolutional Flow”,
*Workshop on Invertible Neural Nets and Normalizing Flows (INNF), ICML 2019* (2019).
- **M. Karami**, L. Dinh, D. Duckworth, J. Sohl-Dickstein, D. Schuurmans, “Generative Convolutional Flow for
Density Estimation”, *Workshop on Bayesian Deep Learning, NeurIPS 2018*, Montreal, Canada (2018).

### Federated Learning and Privacy Preserving ML
- H. Yu, K. Guo, M. Karami, X. Chen, G. Zhang, P. Poupart, “Federated Bayesian Neural Regression: A
Scalable Global Federated Gaussian Process”, *arXiv preprint [arxiv:2206.06357](https://arxiv.org/abs/2206.06357),* (2022).
- M. Hassan, Z. Zhang, k. Guo, M. Karami, G. Zhang, X. Chen, P. Poupart, “Robust One Round Federated
Learning with Predictive Space Bayesian Inference”, *arXiv preprint [arxiv:2206.09526](https://arxiv.org/abs/2206.09526),* (2022).
- D. Jiang, G. Zhang, M. Karami, X. Chen, Y. Shao, Y. Yu, “DP2-VAE: Differentially Private Pre-trained
Variational Autoencoders”, *arXiv preprint [arxiv:2208.03409](https://arxiv.org/abs/2208.03409) arxiv:2208.03409,* (2022).

### Deep Generative Multi-View (Multi-Modal) Learning
<details>
<summary> Abstract </summary>
  We proposed an interpretable deep generative framework for multi-view learning based on a probabilistic formulation of canonical correlation analysis (CCA). The model combines a linear multi-view layer in the latent space with deep generative networks as observation models. The proposed model decomposes the variability between views into a shared latent representation that describes the common underlying sources of variation and a set of view-specific components. We designed an efficient learning algorithm using a variational inference procedure incorporating the solution of probabilistic CCA. This also offered a flexible data fusion method in the latent space. Importantly, the proposed model can be generalized to an arbitrary number of views. An empirical analysis confirms that the proposed deep multi-view model can discover subtle relationships between multiple views and recover rich representations.
</details>

- **M. Karami**, D. Schuurmans, “Deep Probabilistic Canonical Correlation Analysis”, * **AAAI**
conf. on Artificial Intelligence 2021*, available on *arXiv preprint arXiv:2003.04292*, (2021).
[paper](https://papers.nips.cc/paper/2017/hash/c2964caac096f26db222cb325aa267cb-Abstract.html)
[code](https://github.com/Karami-m/Deep-Probabilistic-Multi-View)
- **M. Karami**, “Deep Generative Multi-view Learning”, Accepted for *the workshop Data and Machine Learning
Advances with Multiple Views, ECML/PKDD 2019*, Wurzburg, Germany (2019).

### Sequence Modelling
<details>
<summary> Abstract </summary>
  Maximum likelihood is typically considered to be hard in this setting since latent states and transition parameters must be inferred jointly. Given that expectation-maximization does not scale and is prone to local minima, moment-matching approaches from the subspace identification literature have become standard, despite known statistical efficiency issues. In this work, we instead reconsidered likelihood maximization of LDS with generalized-linear observation models. Key to the approach was a reformulation of the LDS model as a two-view convex optimization problem that allowed us to approximate the estimation task as a form of matrix factorization, and hence apply recent global optimization techniques. Furthermore, a novel proximal mapping update was analytically derived for this two-view reformulation that significantly simplified the optimization procedure. The resulting algorithm was simple to use and flexible enough to incorporate different losses and regularizers while empirical studies demonstrated that this estimation strategy outperforms widely-used identification algorithms such as subspace identification methods, both in terms of accuracy and runtime.
</details>

- **M. Karami**, M. White, D. Schuurmans, C. Szepesvari, “Multi-view Matrix Factorization for Linear Dynamical
System Estimation”, *Advances in Neural Information Processing Systems (**NIPS**)*, (2017).
[paper](https://papers.nips.cc/paper/2017/hash/c2964caac096f26db222cb325aa267cb-Abstract.html)
- **M. Karami**, M. White, D. Schuurmans, “Optimal Linear Dynamical System Identification”, *NIPS Time
Series Workshop, Barcelona, Spain, Dec. 2016* **[Best poster award](https://sites.google.com/site/nipsts2016/)** (2016).

### Wireless Communications
- **M. Karami**, A. Olfat, N. C. Beaulieu, “Pilot Symbol Parameter Optimization Based on Imperfect Channel
State Prediction for OFDM Systems”, *IEEE Transactions on Communications, vol.61, no.6, pp.2557,2567*,
June (2013) [paper](https://ieeexplore.ieee.org/document/6487357).
- **M. Karami** and N. C. Beaulieu, “Channel adaptive power allocation and pilot optimization for OFDM
systems”, *IEEE Global Comm. Conf. (GLOBECOM)*, Anaheim, CA, Dec. (2012) [paper](https://ieeexplore.ieee.org/document/6503894).
- **M. Karami**, A. Olfat and N. C. Beaulieu, “Pilot Symbol Assisted Adaptive Modulation for OFDM Systems
with Imperfect Channel State Information”, in Proc. *IEEE Global Comm. Conf. (GLOBECOM)*, (2010) [paper](https://ieeexplore.ieee.org/document/5683722).
- **M. Karami**, A. Olfat, “Fast Blind Adaptive Channel Shortening Using Signal Subspace”, *IEEE VTC 2008
Spring*, Singapore, pp. 2621-2625, May (2008) [paper](https://ieeexplore.ieee.org/document/4526131).
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
