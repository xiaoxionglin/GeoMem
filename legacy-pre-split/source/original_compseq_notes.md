# Original CompSeq Notes Summary

The original DOCX proposed a research plan called "CompSeq" focused on unifying physical navigation and abstract navigation, especially language, under sequence modeling.

Core original elements:

- Motivation: memory architectures for LLMs, tool use, planning, and reasoning.
- Claim: optimal memory depends on problem geometry.
- Minimal model: Dyck language as tree-like navigation through stack states.
- Geometry control: add shortcuts or commutative equivalences to move from tree-like structure toward Euclidean-like lattices.
- Initial implementation: train an agent to produce grammar-valid symbols with rewards at graph nodes.
- Extension: connect syntax-like navigation to semantic graph navigation.

GeoMem keeps the geometry-dependent memory claim but demotes Dyck language to a limiting case. The new center is task geometry by input density, grounded in hippocampal sequence generators and navigation.

