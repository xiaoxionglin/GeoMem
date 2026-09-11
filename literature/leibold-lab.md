# Leibold Lab Literature Context

## Primary Technical Predecessor

Lin, Yiu, and Leibold 2026 is the central predecessor. It shows that a sparsely driven hippocampus-inspired sequence generator can solve egocentric navigation, outperform LSTMs under sparse input, and produce spatial representations resembling hippocampal phenomena.

GeoMem should explicitly extend this result by manipulating task geometry, not only input sparsity.

## Mechanistic Sequence Foundations

Leibold and Kempter 2006 establishes biological constraints on sequence memory in recurrent networks. Leibold et al. 2008 adds temporal compression through short-term synaptic plasticity. Kammerer, Tejero-Cantero, and Leibold 2013 links inhibition, memory capacity, replay, and oscillations.

## Navigation And Replay

Leibold 2020 proposes that reservoirs of hippocampal sequences can support navigation in unknown environments from sparse landmarks. Chenani et al. 2019 shows that removing MEC input makes CA1 replay less prominent and more rigid, which is important for the interaction between metric input and sequence flexibility.

## Intrinsic And Extrinsic Sequences

Yiu and Leibold 2023 provides a theoretical account of theta correlations that separates extrinsically driven navigation sequences from intrinsic hippocampal sequence structure. GeoMem should use this distinction to avoid claiming that all sequence activity is either sensory-driven or purely cognitive.

## Place/Grid Geometry

Kammerer and Leibold 2014 argues that sparse place coding imposes strong representational constraints. Monsalve-Mercado and Leibold 2017 connects hippocampal spike-timing correlations to hexagonal grid fields. Monsalve-Mercado and Leibold 2020 shows that boundaries can deform grid cell patterns, supporting the claim that neural spatial codes are shaped by environmental geometry.
