---
name: build-docs
description: Build Oceananigans documentation locally with optional fast-build shortcuts
user_invocable: true
---

# Build Documentation

Build the Oceananigans documentation locally.

## Steps

1. Ask the user if they want a full build or a fast build (skipping examples/doctests)
2. For **full build**:
   ```sh
   julia --project=docs/ docs/make.jl
   ```
3. For **fast build**, temporarily modify `docs/make.jl`:
   - Comment out Literate examples in `example_scripts` and `example_pages`
   - Add `warnonly = [:cross_references, :example_block, :linkcheck]`
   - Comment out GPU-requiring pages
   - Optionally set `doctest = false`, `linkcheck = false`, `draft = true`
   - Run the build
   - **Revert all changes** to `docs/make.jl` after building
4. Serve docs locally for preview:
   ```julia
   using LiveServer
   serve(dir="docs/build")
   ```
5. Report any warnings or errors from the build

## Notes

- Fast builds skip examples and doctests - use for checking prose and cross-references only
- Always revert temporary changes to `docs/make.jl` before committing
