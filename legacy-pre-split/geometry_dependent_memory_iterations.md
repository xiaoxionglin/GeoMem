# Additional Iterations on the Geometry-Dependent Memory Architecture Plan

This document extends the earlier five-pass evaluation with five additional iterations. The goal is to sharpen the novelty of the project, identify the strongest literature gap, and propose concrete improvements for a paper about path integration, sequence-based navigation, and task geometry.

## Current central thesis

The best memory architecture depends on task geometry. Mammalian navigation is hybrid because natural environments are often locally metric and approximately Euclidean, but globally graph-like, bottlenecked, hierarchical, and sometimes well described as Gromov-hyperbolic. Path integration is efficient where local displacements commute and loops close reliably. Sequence-based memory is efficient where order, branching, hidden context, or route history matters.

## Iteration 6: Replace "space versus sequence" with "compression regime"

### Evaluation

The current plan still risks sounding like a compromise between two literatures: path integration/grid maps on one side and sequence-centric hippocampus on the other. A stronger framing is that both are compression regimes. Path integration compresses action history into a low-dimensional displacement state. Sequence memory preserves action order when that compression would destroy task-relevant information.

### Existing literature

Grid/path-integration accounts emphasize compact metric updating in entorhinal systems, but recent work questions whether globally regular grid metrics are reliable in realistic environments. Ginosar et al. argue that grid distortions in asymmetric, complex, or three-dimensional environments challenge the idea that grid cells provide a perfect global metric for navigation.

Sequence-centric theories move in the opposite direction. "Space is a latent sequence" argues that spatial representations can emerge from latent higher-order sequence learning, challenging purely space-centric accounts.

### Gap

The field has strong theories for metric compression and for sequence preservation, but less work asks when the task permits history to be compressed without loss. This is the clean normative gap.

### Improvement

Make "lossy versus lossless compression of trajectory history" the formal bridge:

- Euclidean-like geometry: many different paths are equivalent, so history can be compressed into position or displacement.
- Tree-like geometry: different histories often lead to distinct states, so order-preserving memory is required.
- Hybrid geometry: local trajectory segments can be compressed, while global route structure must remain graph-like or sequential.

## Iteration 7: Make commutativity the measurable independent variable

### Evaluation

"Task geometry" is broad. To become experimentally useful, the paper needs at least one measurable quantity that predicts which architecture should work. Commutativity is a strong candidate: whether action sequences AB and BA lead to the same or equivalent state.

### Existing literature

Cognitive graph work already shows that humans do not always rely on globally metric maps. Warren et al.'s "wormholes" experiments support graph-like models that can explain route finding, detours, and rough shortcuts. Recent reviews of cognitive maps and cognitive graphs also argue that map-like and graph-like representations coexist in partially overlapping neural systems.

### Gap

Maps-versus-graphs work often classifies representational format after the fact. It does not usually derive the format from a task-level algebraic property such as action commutativity.

### Improvement

Define an "effective commutativity index" for toy environments:

- Sample pairs of action strings with the same multiset of actions.
- Measure whether they terminate in the same state, nearby states, or task-equivalent states.
- Predict that high commutativity favors coordinate/path-integration memory, while low commutativity favors sequence, stack, or clone-state memory.

This turns the theory from a verbal geometry argument into a testable computational claim.

## Iteration 8: Treat local Euclidean patching as a key biological constraint

### Evaluation

The phrase "locally flat, globally hyperbolic" should be connected to behavioral evidence. The best recent support is not that humans perfectly represent non-Euclidean spaces, but that they often use locally Euclidean approximations and stitch them together.

### Existing literature

Kim and Doeller studied path integration and spatial memory on spherical versus planar environments in immersive VR. They found a strong Euclidean bias on the sphere, but also reasonable navigation performance, suggesting that humans can navigate non-flat surfaces by constructing locally confined Euclidean maps and flexibly combining them.

Mou's 2025 review emphasizes that human navigation uses place knowledge, route knowledge, and survey knowledge, and that environmental variables affect when cognitive maps form.

### Gap

There is a gap between behavioral evidence for local Euclidean patching and computational theories of how memory architecture should change across local versus global scale.

### Improvement

Add a scale-dependent claim:

- At short ranges, path integration can operate inside locally Euclidean patches.
- Across patch boundaries, memory must track transitions, landmarks, routes, and graph topology.
- The hippocampal-entorhinal system should therefore show architecture switching or mixture-of-experts behavior as a function of navigational scale.

This gives the theory a biological reason for being hybrid, not merely a computational one.

## Iteration 9: Position against successor representation and TEM more precisely

### Evaluation

The previous plan mentions successor representation and the Tolman-Eichenbaum Machine, but it should say exactly what this project adds. Otherwise reviewers may see it as a restatement of predictive maps or structural generalization.

### Existing literature

Stachenfeld, Botvinick, and Gershman frame the hippocampus as a predictive map: place-cell-like representations can encode expected future occupancy and are shaped by policy and reward, not only geometry.

The Tolman-Eichenbaum Machine explains how the hippocampal formation can generalize relational structure across tasks by separating structural representations from sensory content.

Recent hippocampal scaffold work argues that grid-cell-like states can support both spatial maps and high-capacity sequence memory by reducing sequence learning to low-dimensional transitions.

### Gap

SR and TEM describe powerful representational objectives: prediction, generalization, and structure learning. They do not fully specify when the memory substrate should be vector-like, graph-like, stack-like, episodic, or sequential as a function of environmental geometry.

### Improvement

State the relationship explicitly:

- SR answers: what future states are expected under a policy?
- TEM answers: how can structural knowledge generalize across domains?
- The proposed theory answers: what memory architecture is efficient for a given task geometry?

Then use SR/TEM as compatible mechanisms that may implement different regions of the geometry-memory phase space.

## Iteration 10: Add AI-agent memory as a second test bed, not only motivation

### Evaluation

The original plan mentions LLM memory architectures but does not yet exploit them as a research test bed. This is a missed opportunity because AI-agent memory systems now actively compare vector, graph, hierarchical, and episodic memories.

### Existing literature

Recent agent-memory systems increasingly move beyond flat context windows and vector retrieval. Task Memory Engine uses graph-like task memory for multi-step LLM agents. AriGraph uses knowledge graph world models with episodic memory for interactive text-game agents. Mem4Nav combines metric-like spatial memory with semantic topology graphs for urban vision-language navigation.

### Gap

AI-agent memory work is mostly engineering-driven: graph memory, vector memory, hierarchy, and episodic traces are compared by benchmark performance, but rarely by a principled account of task geometry.

### Improvement

Add an explicit AI evaluation track:

- Construct benchmark families whose geometry is controlled: lattice worlds, trees, shortcut graphs, mazes with bottlenecks, and Dyck-like symbolic navigation.
- Compare vector retrieval, graph memory, recurrent sequence memory, stack-augmented memory, and hybrid memory.
- Ask whether the same geometry-to-architecture rule predicts both biological navigation behavior and AI-agent memory performance.

This makes the paper more broadly relevant and gives it a practical validation path.

## Refined gap statement

Existing literature already argues that navigation can use metric maps, graph-like route structures, predictive maps, structural generalization, sequence learning, and hippocampal scaffolds. The missing piece is a normative and testable account of when each memory architecture is optimal. The proposed contribution is to make task geometry the independent variable: especially local flatness, global graph structure, bottlenecks, branching factor, path equivalence, and action commutativity.

## Revised paper title options

1. Task Geometry Determines Memory Architecture for Navigation
2. From Path Integration to Sequence Memory: Geometry as the Missing Variable
3. Why Mammalian Navigation Is Hybrid: Local Euclidean Patches in Global Graph Spaces
4. Commutativity, Compression, and Cognitive Maps

## Recommended revised abstract

Navigation requires memory, but no single memory architecture is optimal across all environments. We propose that the appropriate architecture is determined by task geometry. In locally Euclidean regions, action sequences can be compressed into coordinate-like displacement states, supporting path integration and vector navigation. In globally graph-like, bottlenecked, hierarchical, or Gromov-hyperbolic environments, action order remains task-relevant, requiring sequence, stack, graph, or clone-state memory. Mammalian navigation is therefore expected to be hybrid: locally metric but globally route- and sequence-sensitive. We formalize this claim using a family of environments that interpolate from Dyck trees to shortcut-augmented graphs and Euclidean lattices, with action commutativity as a measurable bridge variable. The framework explains why path integration, cognitive graphs, predictive maps, structural generalization, and sequence-based hippocampal theories each capture part of the truth, while also predicting when each should fail.

## Concrete improvements to the plan

- Replace the opposition between path integration and sequence navigation with a compression tradeoff: when can trajectory history be safely compressed?
- Use effective commutativity as the main measurable variable connecting symbolic sequence tasks and spatial navigation.
- Emphasize local Euclidean patching as the biological interpretation of "locally flat."
- Position SR and TEM as compatible mechanisms, not competitors.
- Add an AI-agent benchmark track to test whether geometry predicts memory architecture outside neuroscience.
- Avoid claiming that natural environments are literally globally hyperbolic unless the paper defines this operationally. Prefer "globally graph-like or Gromov-hyperbolic at behavioral scales."

## Key references

- Peer et al., 2021. "Structuring Knowledge with Cognitive Maps and Cognitive Graphs." https://pmc.ncbi.nlm.nih.gov/articles/PMC7746605/
- Warren et al., 2017. "Wormholes in virtual space: From cognitive maps to cognitive graphs." https://pubmed.ncbi.nlm.nih.gov/28577445/
- Mou, 2025. "Representing place locations and orientations in cognitive maps." https://www.nature.com/articles/s44159-025-00442-0
- Kim and Doeller, 2024. "Cognitive Maps for a Non-Euclidean Environment: Path Integration and Spatial Memory on a Sphere." https://journals.sagepub.com/doi/abs/10.1177/09567976241279291
- Ginosar et al., 2023. "Are grid cells used for navigation? On local metrics, subjective spaces, and black holes." https://www.weizmann.ac.il/brain-sciences/labs/ulanovsky/sites/brain-sciences.labs.ulanovsky/files/2024-11/Ginosar2023a.pdf
- Stachenfeld, Botvinick, and Gershman, 2017. "The hippocampus as a predictive map." https://www.nature.com/articles/nn.4650
- Whittington et al., 2020. "The Tolman-Eichenbaum Machine." https://discovery.ucl.ac.uk/id/eprint/10115119/
- George et al., 2024. "Space is a latent sequence: A theory of the hippocampus." https://www.ovid.com/journals/sciad/abstract/10.1126/sciadv.adm8470~space-is-a-latent-sequence-a-theory-of-the-hippocampus
- Zhang et al., 2022. "Hippocampal spatial representations exhibit a hyperbolic geometry that expands with experience." https://www.nature.com/articles/s41593-022-01212-4
- Chandra et al., 2025. "Episodic and associative memory from spatial scaffolds in the hippocampus." https://www.nature.com/articles/s41586-024-08392-y
- Task Memory Engine, 2025. "Spatial Memory for Robust Multi-Step LLM Agents." https://arxiv.org/abs/2505.19436
- AriGraph, 2025. "Learning Knowledge Graph World Models with Episodic Memory for LLM Agents." https://www.ijcai.org/proceedings/2025/0002
- Mem4Nav, 2025. "Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System." https://huggingface.co/papers/2506.19433

## Professor Evaluation: Computational Neuroscience Framing

### Overall evaluation

The plan is promising and intellectually coherent, but the strongest version is not a grand unifying theory of all navigation. It is a computational theory of when hippocampal sequence generators are useful.

The tightest framing is:

> Intrinsic sequence dynamics are useful when task geometry and sensory sparsity make current state hard to infer from local input; path-integration/vector memory is useful when action history is compressible into low-dimensional coordinates.

This is a natural extension of Lin, Yiu, and Leibold, ICLR 2026, where sparse sensory evidence favors a hippocampus-inspired sequence generator.

### Scientific strengths

- The plan has strong lab continuity with work on hippocampal sequence reservoirs, theta sequences, replay/preplay, sparse coding, grid/place geometry, and navigation.
- The conceptual gap is real: existing work describes maps, graphs, successor representations, TEM, and sequence models, but less work asks when each architecture is optimal.
- The computational direction is testable because environment geometry and input sparsity can be manipulated independently.
- The ICLR 2026 agent is an excellent baseline because it already provides a concrete architecture, task, and biological interpretation.

### Main weakness

The plan risks being too broad. "Task geometry determines memory architecture" is elegant, but it needs operational definitions. Otherwise reviewers may read geometry as a metaphor rather than as an experimental variable.

The primary measurable axes should be limited to a small set:

- input sparsity;
- action commutativity;
- sensory aliasing;
- loop closure reliability;
- bottleneck and branching structure.

Graph hyperbolicity and local metric consistency can remain secondary or analytic measures, but they should not all be introduced as equally central.

### Recommended revisions

Make the core model a two-dimensional hypothesis space: task geometry by input density.

```text
                          Input density
                    sparse             dense
geometry
metric              sequence may help   path integration/vector state
tree-like           sequence            sequence or graph
mixed               hybrid              hybrid or graph
```

This matters because Lin, Yiu, and Leibold 2026 shows that sequence generators help under sparse input, but not dense input. GeoMem should generalize that result by asking whether geometry modulates this sparsity effect.

Use this tightened thesis:

> Hippocampal sequence dynamics are not universally optimal memory mechanisms. They are advantageous when sparse observations and noncommutative task geometry prevent compression of action history into a low-dimensional metric state. Path-integration and vector memory dominate when local metric structure is reliable and action histories commute. Mammalian navigation is hybrid because natural environments combine locally metric regions with globally graph-like structure.

### Partial commutation

Partial commutation is the best formal idea, but it should not dominate the main exposition. It gives the right theory: if actions commute, history can collapse into coordinates; if they do not, order matters and sequence memory is needed.

For neuroscience readers, the main text should explain it in plain terms:

> In some environments, doing A then B is equivalent to doing B then A. In others, order changes the state. The more order matters, the more memory must preserve sequence history.

Trace monoids or RAAG-like formalisms can remain a formal scaffold or appendix. The main paper should use the intuition to generate concrete experiments and predictions.

### Dyck language

Dyck language should remain a limiting case only. It is useful for stack memory, hierarchy, and long-range dependency, but it is not a good central navigation model. If Dyck becomes central, the paper will look like a formal-language paper rather than a computational neuroscience theory.

### Leibold lab continuity

The lab-aligned version is strong. The project should be framed as:

> From "hippocampal sequences help sparse-input navigation" to "the usefulness of hippocampal sequences is predictable from task geometry and input statistics."

Relevant lab continuity:

- Lin, Yiu, and Leibold 2026 shows that a hippocampus-inspired sequence generator can support egocentric visual navigation under sparse sensory evidence.
- Leibold 2020 provides a sequence-reservoir model for navigation in unknown environments.
- Yiu and Leibold 2023 distinguishes intrinsic and extrinsic sources of hippocampal theta sequences.
- Chenani et al. 2019 shows that medial entorhinal inputs affect the flexibility of hippocampal replay.
- Kammerer and Leibold 2014 links sparse place coding constraints to representational capacity.
- Monsalve-Mercado and Leibold 2017/2020 connects hippocampal correlations, grid fields, and environmental geometry.

The missing step is systematic: when should sequence dynamics be preferred over coordinate, vector, or path-integration memory?

### Reviewer risks

- Avoid overclaiming biology. Present the theory as a computational interpretation, not a one-to-one anatomical mapping.
- Avoid letting modularity make the paper too broad. The paper itself needs one clean path, even if the planning files remain modular.
- Define "optimal" explicitly in terms of sample efficiency, generalization, robustness, and memory cost.
- Distinguish clearly from successor representation, TEM, cognitive graphs, and sequence-centric hippocampal theories.
- Make sure partial commutation produces experimental predictions rather than mathematical elegance alone.

### Final judgment

This is a strong project if narrowed. It should not be pitched first as a grand unifying theory of all navigation. It should be pitched as a computational theory of when hippocampal sequence generators are useful, grounded in the Leibold lab's existing models, and then broadened to memory architecture and task geometry.
