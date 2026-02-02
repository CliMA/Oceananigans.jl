# FFT AD Investigation Tests

**Investigation:** FFT Compilation / Differentiation (B.6.1)  
**Related:** `cursor-toolchain/rules/domains/differentiability/investigations/fft-compilation.md`  
**Synchronized with:** `Manteia.jl/test/fft-ad/` (downstream)

## Current Status

| Component | Status |
|-----------|--------|
| FFT plan extension (`Solvers.jl`) | ✅ Implemented |
| Periodic topology (FFT) | ✅ Works |
| Bounded topology (DCT) | ❌ Not supported (clear error) |
| RK3 timestepper | ❌ Not Reactant-compatible (blocks AD) |

## Test Files

| File | Purpose | Status |
|------|---------|--------|
| `test_fft_compilation.jl` | Model construction tests | ✅ Passing |
| `test_fft_medwe.jl` | Full AD pipeline | ❌ Blocked by RK3 |

## How to Run

```julia
cd("/Users/danielkz/Aeolus2/Oceananigans.jl/test")
# julia --project=.
include("fft-ad/test_fft_compilation.jl")  # Model construction
include("fft-ad/test_fft_medwe.jl")         # Full AD (blocked)
```

## Next Steps

1. Sync fork with upstream Oceananigans (may have RK3 support)
2. If needed, add RK3 Reactant support to `TimeSteppers.jl`
