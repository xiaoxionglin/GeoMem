---
tags:
  - geomem
  - internal-summary
  - collaboration
status: draft
date: 2026-07-14
---

# GeoMem Project Idea Summary

## One-Sentence Pitch

GeoMem asks when navigation and control histories can be compressed into compact current state, and when task geometry, sparse observations, or latent route context make explicit memory necessary.

## Core Idea

The project treats memory demand as an interaction between three factors: task geometry, observation quality, and behavioral goal. In locally metric, well-observed environments, many action histories collapse into equivalent state; vector-like or coordinate-like memory can be enough. In trees, bottlenecks, route-choice tasks, aliased rooms, or cue-gap tasks, the same current observation can hide distinct histories that require different future actions. Those regimes create a computational need for sequence memory, finite history, reservoir dynamics, or hybrid representations.

The key framing is not that one memory architecture is generally best. The claim is conditional: the appropriate memory system depends on which history variables remain behaviorally relevant after the task and observation stream have compressed everything else.

## Innovative Parts

- The project operationalizes history compressibility with measurable diagnostics: effective commutativity, observation ambiguity, transition ambiguity, policy ambiguity, and residual-history pressure.
- It separates latent task geometry, observation mode, and model memory, making architecture comparisons interpretable rather than opaque benchmark outcomes.
- It connects formal trace-monoid environments to navigation-like rooms, bottlenecks, cue gaps, and route choices.
- It tests a hybrid-memory interpretation: local metric information may be compressible while global route or latent-context information still requires explicit memory.
- It treats hippocampal sequence-like memory, path-integration/vector memory, reservoirs, RNNs, and finite-history controls as different solutions to different geometry-induced memory problems.

## Why This Is Publication-Relevant

The publication angle is a mechanistic account of memory architecture selection. Instead of claiming that sequence memory, path integration, or reservoirs are universally superior, GeoMem predicts when each should become useful.

The currently defensible claim is:

> Coordinate/vector memory is sufficient when task geometry makes action order irrelevant. When action order remains behaviorally relevant under aliased or sparse observations, sequence-like or hybrid memory becomes useful.

This claim is relevant to computational neuroscience, cognitive maps, hippocampal-entorhinal function, and machine-learning benchmarks for memory because it links architectural needs to measurable properties of the task.

## Defining Features

- Task geometry is the organizing variable: commutativity, bottlenecks, branching, aliasing, loop structure, and cue visibility are manipulated explicitly.
- Observation modes are explicit: full, count/vector-like, sparse aliased, egocentric landmark-like, Gaussian sensory, and hybrid observations are treated as different current-state interfaces.
- Memory is model-side: the environment emits current observation `o_t`; memoryless models, RNNs, reservoirs, Lin sequence bases, finite histories, and hybrids differ in how they preserve prior information.
- Diagnostics precede agents: the project first asks whether observations are Markov, whether they alias policies, and whether history is theoretically required.
- Architecture comparisons are conditional: the important result is that the best memory mechanism changes with geometry and observation regime.

## Current Internal Position

GeoMem is best described as a theory-plus-benchmark project for geometry-dependent memory. Its near-term publication strength is not full biological realism or reinforcement-learning agent superiority. The strongest current contribution is about representational sufficiency: when task geometry and observation structure make current state insufficient, different memory architectures become necessary in predictable ways.
