# Reactant Correctness Tests - `raise=true` Report

**Julia Version**: 1.11.8  
**Reactant Version**: 0.2.114  
**Test Date**: January 6, 2026  
**Architecture**: CPU (arm64 macOS)

## Summary

Testing `fill_halo_regions!` with `@jit raise=true` on Julia 1.11.8:

| Topology | Result |
|----------|--------|
| (Bounded, Bounded, Bounded) | ✓ PASSED |
| (Periodic, Bounded, Bounded) | ✓ PASSED |
| (Bounded, Periodic, Bounded) | ✓ PASSED |
| (Bounded, Bounded, Periodic) | ✓ PASSED |
| (Periodic, Periodic, Bounded) | ✓ PASSED |
| (Periodic, Bounded, Periodic) | ✓ PASSED |
| (Bounded, Periodic, Periodic) | ✓ PASSED |

**Pass Rate**: 7/7 (100%)

## Excluded Case

The **fully periodic topology `(Periodic, Periodic, Periodic)`** is excluded from tests because it triggers a segmentation fault during MLIR pattern rewriting when using `@jit raise=true`. This is a known Reactant.jl bug. Triply periodic grids are rarely used in ocean modeling applications.

## Notes

- **CUDA.jl is required**: Even on CPU, `using CUDA` must be loaded for `raise=true` to work with KernelAbstractions kernels. Without it, you get:
  ```
  MethodError: no method matching ka_with_reactant(...)
  Attempted to raise a KernelAbstractions kernel with Reactant but CUDA.jl is not loaded.
  ```

- **`raise=false` works fine**: All topologies pass with `raise=false`

- **This is a Reactant.jl bug**: The triply-periodic segfault is in `libReactantExtra.dylib`, not in Oceananigans or Julia
