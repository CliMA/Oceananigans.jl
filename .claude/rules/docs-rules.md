---
paths:
  - docs/**/*
---

# Documentation Rules

Building the docs, including a fast local build, is covered by the `/build-docs` skill
(`.claude/skills/build-docs/SKILL.md`).

## Style

- Use Documenter.jl syntax for cross-references
- Add paper references via bibtex in `oceananigans.bib` with corresponding citations
- Make use of cross-references with equations
- In example code, rely on `using Oceananigans`; explicitly importing an exported name hides
  what users actually need to type

Docstring conventions are in `.claude/rules/docstring-rules.md`.
