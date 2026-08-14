## higher level critiques

we should prune the variants of environments and architectures? Are many of them not informative? Is there some representative things missing? for example, is there too much emphasis on the T-maze?

There are a lot of results which is hard for me to understand in a short amount of time, which I assume would be the same when the paper is presented to the audience/reviewers. We should improve the crispness of the communication, improve the higher-level ordering of ideas.

## Opinion 1 (answered):

This is a note for mostly high level evaluation from the admin. You can reply below and argue for the best approach.

The way I see it now, the action history framing serves as a good starter for formalizing the "geometry" of the environment. We need to strengthen this point, actually form a measure that could be used on a broader class of realistic environments, e.g. deepmind lab, web agent benchmarks, grid world, etc. There's also the possible addition to the core thesis that it's not jsut the hybrid system fits a mid-level commutativity, but maybe the hybrid memory system is optimal when considering the performance on a variety of environments together.

On the other hand, to form a theory of the real world / real organism, it seems to me the action history is similar to the path-integration system, but the majority of the input for realistic systems is a separate sensory input. If this is a coherent narrative, the framing in the paper should make this distinction clear, that the action-history mainly serves as a measure for the geometry.

## Response

I agree with the separation, with one qualification: action history is not itself the path-integration system. It is the path object on which the geometry acts. A path-integration-like system is one possible compression of that path into a local metric state. The formal trace tasks should therefore be presented as a controlled geometry probe: they ask which action histories the task identifies as equivalent, and which residual order distinctions survive that quotient. They are not a model of the dominant sensory stream in realistic navigation.

That distinction should become explicit in the paper:

- Geometry axis: action-history equivalence, local reorderability, loop closure, bottlenecks, and task-relevant path compression.
- Sensory axis: the observation-emission channel from latent task state to visual, symbolic, noisy, aliased, persistent, cue-gap, or DG-sparsified input.
- Memory axis: the representation or mechanism that retains the geometry- and observation-dependent residual information needed for control.

The action-history framing remains useful because it can lead to geometry measures that generalize beyond the trace family. The immediate target should not be a single "commutativity" number claimed to characterize DeepMind Lab, web agents, and grid worlds equally well. The better target is a small diagnostic family:

1. Local action reorderability: when two valid short action blocks are swapped, how often do they reach equivalent latent or task states?
2. Path-compression gap: how much policy- or transition-relevant ambiguity remains after a coordinate-like path summary such as displacement, action counts, or a learned local integrator state?
3. Route residual: how much control-relevant information must be retained after the local summary is supplied?

On grid worlds and simulator environments these can use privileged latent state, task state, or exact transition graphs. In visual environments such as DeepMind Lab they should be computed first with simulator state and task outcomes, then compared against what can be recovered from sensory emissions. In web-agent benchmarks, "state equivalence" must be defined by task-relevant browser state or downstream action-value equivalence rather than by pixel or DOM similarity alone.

The cross-environment hybrid claim is plausible and stronger than the current mid-regime statement, but it should be stated as a distributional hypothesis until it is tested. A hybrid architecture may be the best fixed memory choice over a varied task family because it has lower regret across locally metric, route-dependent, and mixed environments. That claim requires an explicit task distribution, cost budget, and comparison against specialized vector-only and sequence-only systems. It is a good extension because mammals face a task portfolio, not a single commutativity level.

## Recommended Framing Change

The paper should lead with:

> Action-history quotients formalize the geometry of control paths. Sensory emissions determine what evidence about the latent task state is currently available. Memory demand is the residual task-relevant history that remains after both path compression and sensory evidence are accounted for.

The present trace results establish the first component cleanly. The room-graph and delayed-cue experiments start to couple it to observation emissions. A realistic DeepMind Lab extension should test the same separation with visual observations and a DG-like front end.
