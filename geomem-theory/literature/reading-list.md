# GeoMem Reading List

## Primary Lab Anchors

- Lin, X.-X., Yiu, Y.-H., & Leibold, C. (2026). Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator. ICLR 2026. https://openreview.net/forum?id=li1vfqDzRD and https://doi.org/10.48550/arXiv.2510.09951  
  Role: main technical predecessor; sparse sensory evidence interacts with sequence-generator memory architecture.

- Leibold, C. (2020). A model for navigation in unknown environments based on a reservoir of hippocampal sequences. Neural Networks. https://doi.org/10.1016/j.neunet.2020.01.014  
  Role: sequence reservoir for sparse-landmark navigation.

- Yiu, Y.-H., & Leibold, C. (2023). A theory of hippocampal theta correlations accounting for extrinsic and intrinsic sequences. eLife. https://doi.org/10.7554/eLife.86837.4  
  Role: distinction between behavior-driven and intrinsic sequence structure.

- Chenani, A., Sabariego, M., Schlesiger, M. I., Leutgeb, J. K., Leutgeb, S., & Leibold, C. (2019). Hippocampal CA1 replay becomes less prominent but more rigid without inputs from medial entorhinal cortex. Nature Communications. https://doi.org/10.1038/s41467-019-09280-0  
  Role: MEC input affects replay flexibility.

- Leibold, C., & Kempter, R. (2006). Memory capacity for sequences in a recurrent network with biological constraints. Neural Computation. https://doi.org/10.1162/089976606775774714  
  Role: mechanistic foundation for sequence-memory capacity.

- Leibold, C., Gundlfinger, A., Schmidt, R., Thurley, K., Schmitz, D., & Kempter, R. (2008). Temporal compression mediated by short-term synaptic plasticity. PNAS. https://doi.org/10.1073/pnas.0708711105  
  Role: bridge between behavioral sequences and compressed replay.

- Kammerer, A., Tejero-Cantero, A., & Leibold, C. (2013). Inhibition enhances memory capacity: optimal feedback, transient replay and oscillations. Journal of Computational Neuroscience.  
  Role: recurrent sequence memory and replay mechanism. DOI should be verified before citation.

- Kammerer, A., & Leibold, C. (2014). Hippocampal remapping is constrained by sparseness rather than capacity. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1003986  
  Role: sparse coding constraint relevant to sparse-input sequence advantage.

- Monsalve-Mercado, M. M., & Leibold, C. (2017). Hippocampal spike-timing correlations lead to hexagonal grid fields. Physical Review Letters. https://doi.org/10.1103/PhysRevLett.119.038101  
  Role: correlation dynamics can produce grid-like structure.

- Monsalve-Mercado, M. M., & Leibold, C. (2020). Effect of boundaries on grid cell patterns. Physical Review Research. https://doi.org/10.1103/PhysRevResearch.2.043137  
  Role: environmental geometry deforms grid codes.

## Task Geometry, Complexity, and Environment Measures

- Gao, P., & Ganguli, S. (2015). On simplicity and complexity in the brave new world of large-scale neuroscience. Current Opinion in Neurobiology. https://doi.org/10.1016/j.conb.2015.04.003  
  Role: **first priority for the environment/measure work.** Neural task complexity (NTC) bounds representational dimensionality using the volume of the task-parameter manifold and the neural correlation length. The volume-over-correlation-volume construction is the closest existing template for a curvature-like GeoMem task index, but GeoMem should define its index on the controlled task and observation process rather than on a trained neural representation.

- Gao, P., Trautmann, E., Yu, B. M., Santhanam, G., Ryu, S., Shenoy, K. V., & Ganguli, S. (2017). A theory of multineuronal dimensionality, dynamics and measurement. bioRxiv. https://doi.org/10.1101/214262  
  Role: technical follow-up for NTC, including correlation-volume formulas and sampling predictions. Useful for separating intrinsic task complexity from apparent dimensionality caused by a particular model or recording.

- Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. Journal of Functional Analysis. https://doi.org/10.1016/j.jfa.2008.11.001  
  Role: principled local curvature for stochastic transition kernels, defined by contraction or expansion of nearby one-step distributions in Wasserstein distance. A strong candidate diagnostic for environments, but it must be tested against GeoMem's commutativity, transition ambiguity, and policy ambiguity rather than assumed to predict memory demand by itself.

- Zhang, H., Rich, P. D., Lee, A. K., & Sharpee, T. O. (2023). Hippocampal spatial representations exhibit a hyperbolic geometry that expands with experience. Nature Neuroscience. https://doi.org/10.1038/s41593-022-01212-4  
  Role: empirical motivation for negative curvature and hierarchical/tree-like codes. Especially relevant because the paper explicitly connects hyperbolic noncommutativity to spike order, but its inferred representational curvature is not yet a task-intrinsic classifier.

- Urdapilleta, E., Troiani, F., Stella, F., & Treves, A. (2015). Can rodents conceive hyperbolic spaces? Journal of the Royal Society Interface. https://doi.org/10.1098/rsif.2014.1214  
  Role: key counterexample and environment-design guide: grid-like organization can remain locally viable in negatively curved space when grid scale is small relative to curvature radius. Motivates sweeping the dimensionless path-length/curvature-radius ratio instead of treating Euclidean and hyperbolic tasks as a binary contrast.

- Hardcastle, K., Ganguli, S., & Giocomo, L. M. (2015). Environmental boundaries as an error correction mechanism for grid cells. Neuron. https://doi.org/10.1016/j.neuron.2015.03.039  
  Role: shows how boundaries and sensory correction interact with path integration. It motivates treating boundary conditions and cue availability as independent environment factors, not as incidental implementation details.

## Nearby Theories

- McNaughton, B. L., Battaglia, F. P., Jensen, O., Moser, E. I., & Moser, M.-B. (2006). Path integration and the neural basis of the cognitive map. Nature Reviews Neuroscience. https://doi.org/10.1038/nrn1932

- Chrastil, E. R., & Warren, W. H. (2014). From cognitive maps to cognitive graphs. PLOS ONE. https://doi.org/10.1371/journal.pone.0112544

- Peer, M., Brunec, I. K., Newcombe, N. S., & Epstein, R. A. (2021). Structuring knowledge with cognitive maps and cognitive graphs. Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2020.10.004

- Baumann, T., & Mallot, H. A. (2023). Metric information in cognitive maps: Euclidean embedding of non-Euclidean environments. PLOS Computational Biology. https://doi.org/10.1371/journal.pcbi.1011748

- Stachenfeld, K. L., Botvinick, M. M., & Gershman, S. J. (2017). The hippocampus as a predictive map. Nature Neuroscience. https://doi.org/10.1038/nn.4650

- Whittington, J. C. R., et al. (2020). The Tolman-Eichenbaum Machine: Unifying space and relational memory through generalization in the hippocampal formation. Cell. https://doi.org/10.1016/j.cell.2020.10.024

- George, D., et al. (2021). Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. Nature Communications. https://www.nature.com/articles/s41467-021-22559-5

- George, D., et al. (2024). Space is a latent sequence: A theory of the hippocampus. Science Advances. https://doi.org/10.1126/sciadv.adm8470
  Role: **user-supplied reading.** Sequence-first account in which spatial structure is learned from sensory sequences; a direct contrast to GeoMem's hypothesis that sequence memory is selected only in some geometry-observation regimes.

- Cueva, C. J., & Wei, X.-X. (2018). Emergence of grid-like representations by training recurrent neural networks to perform spatial localization. ICLR 2018. https://doi.org/10.48550/arXiv.1803.07770  
  Role: **user-supplied reading.** Canonical Euclidean RNN path-integration baseline: velocity-driven localization yields grid-, border-, and band-like units. The student's environments should reproduce this regime before interpreting curvature-dependent changes.

- Wang, Z., Di Tullio, R. W., Rooke, S., & Balasubramanian, V. (2024). Time Makes Space: Emergence of Place Fields in Networks Encoding Temporally Continuous Sensory Experiences. NeurIPS 2024. https://arxiv.org/abs/2408.05798  
  Role: **user-supplied reading.** Temporally continuous masked sensory reconstruction yields place-like fields, remapping, drift, and continual recall without spatial supervision. It motivates sensory continuity, room identity, revisit structure, and masking/cue loss as explicit environment axes.

- Wang, Z., Morris, G., Derdikman, D., Chaudhari, P., & Balasubramanian, V. (2026). A simple model of co-emergence of grid and place fields. arXiv. https://doi.org/10.48550/arXiv.2605.21356  
  Role: recent unified recurrent sensory-prediction model in which grid and place fields co-emerge without supervising either representation; useful contrast for GeoMem's focus on task geometry and memory-substrate selection.

- Banino, A., et al. (2018). Vector-based navigation using grid-like representations in artificial agents. Nature. https://doi.org/10.1038/s41586-018-0102-6

- Suzgun, M., Gehrmann, S., Belinkov, Y., & Shieber, S. M. (2019). Memory-Augmented Recurrent Neural Networks Can Learn Generalized Dyck Languages. arXiv. https://doi.org/10.48550/arXiv.1911.03329

- Hewitt, J., et al. (2020). RNNs can generate bounded hierarchical languages with optimal memory. arXiv. https://arxiv.org/abs/2010.07515
