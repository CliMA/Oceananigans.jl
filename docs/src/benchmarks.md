# [Performance benchmarks](@id performance_benchmarks)

The performance benchmarking scripts in the `benchmarks` directory of the git repository
can be run to benchmark Oceananigans.jl on your machine. They use
[TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl) to nicely format the
benchmarks.

## Static ocean

This is a benchmark of a simple "static ocean" configuration. The time stepping and Poisson
solver still takes the same amount of time whether the ocean is static or active, so it should
be indicative of actual performance. It tests the performance of a bare-bones model.

```
Julia Version 1.3.0
Commit 46ce4d7933 (2019-11-26 06:09 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, broadwell)
  GPU: Tesla V100-PCIE-32GB

 ──────────────────────────────────────────────────────────────────────────────────────
        Static ocean benchmarks                Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 153s / 77.9%           7.36GiB / 0.91%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32  (CPU, Float32)       10   78.7ms  0.07%  7.87ms    768KiB  1.09%  76.8KiB
  32× 32× 32  (CPU, Float64)       10   79.0ms  0.07%  7.90ms    768KiB  1.09%  76.8KiB
  32× 32× 32  (GPU, Float32)       10   41.3ms  0.03%  4.13ms   7.83MiB  11.4%   802KiB
  32× 32× 32  (GPU, Float64)       10   42.6ms  0.04%  4.26ms   7.84MiB  11.4%   803KiB
  64× 64× 64  (CPU, Float32)       10    685ms  0.58%  68.5ms    768KiB  1.09%  76.8KiB
  64× 64× 64  (CPU, Float64)       10    674ms  0.57%  67.4ms    768KiB  1.09%  76.8KiB
  64× 64× 64  (GPU, Float32)       10   44.1ms  0.04%  4.41ms   7.84MiB  11.4%   802KiB
  64× 64× 64  (GPU, Float64)       10   43.4ms  0.04%  4.34ms   7.84MiB  11.4%   803KiB
 128×128×128  (CPU, Float32)       10    5.72s  4.82%   572ms    768KiB  1.09%  76.8KiB
 128×128×128  (CPU, Float64)       10    5.59s  4.70%   559ms    768KiB  1.09%  76.8KiB
 128×128×128  (GPU, Float32)       10   54.0ms  0.05%  5.40ms   7.84MiB  11.4%   802KiB
 128×128×128  (GPU, Float64)       10   54.6ms  0.05%  5.46ms   7.84MiB  11.4%   803KiB
 256×256×256  (CPU, Float32)       10    54.3s  45.7%   5.43s    768KiB  1.09%  76.8KiB
 256×256×256  (CPU, Float64)       10    50.8s  42.8%   5.08s    768KiB  1.09%  76.8KiB
 256×256×256  (GPU, Float32)       10    305ms  0.26%  30.5ms   7.84MiB  11.4%   802KiB
 256×256×256  (GPU, Float64)       10    303ms  0.26%  30.3ms   7.84MiB  11.4%   803KiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.004
 64× 64× 64 : 0.985
128×128×128 : 0.976
256×256×256 : 0.936

GPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.031
 64× 64× 64 : 0.985
128×128×128 : 1.012
256×256×256 : 0.994

CPU -> GPU speedup:
 32× 32× 32  (Float32): 1.904
 32× 32× 32  (Float64): 1.853
 64× 64× 64  (Float32): 15.531
 64× 64× 64  (Float64): 15.527
128×128×128  (Float32): 106.054
128×128×128  (Float64): 102.323
256×256×256  (Float32): 177.938
256×256×256  (Float64): 167.630
```

## Eddying channel

This benchmark tests the channel model configuration which can be slower due to the use of
a more complicated algorithm for the pressure solver in the current version of Oceananigans.

```
Julia Version 1.3.0
Commit 46ce4d7933 (2019-11-26 06:09 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, broadwell)
  GPU: Tesla V100-PCIE-32GB

 ──────────────────────────────────────────────────────────────────────────────────────
       Eddying channel benchmarks              Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 112s / 61.5%           9.67GiB / 0.38%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32  (CPU, Float32)        5   45.1ms  0.07%  9.03ms    389KiB  1.02%  77.8KiB
  32× 32× 32  (CPU, Float64)        5   48.4ms  0.07%  9.68ms    389KiB  1.02%  77.8KiB
  32× 32× 32  (GPU, Float32)        5   33.1ms  0.05%  6.62ms   4.07MiB  10.9%   834KiB
  32× 32× 32  (GPU, Float64)        5   32.1ms  0.05%  6.42ms   4.08MiB  10.9%   835KiB
  64× 64× 64  (CPU, Float32)        5    377ms  0.55%  75.5ms    389KiB  1.02%  77.8KiB
  64× 64× 64  (CPU, Float64)        5    379ms  0.55%  75.7ms    389KiB  1.02%  77.8KiB
  64× 64× 64  (GPU, Float32)        5   44.7ms  0.06%  8.93ms   4.15MiB  11.1%   850KiB
  64× 64× 64  (GPU, Float64)        5   44.1ms  0.06%  8.82ms   4.15MiB  11.1%   850KiB
 128×128×128  (CPU, Float32)        5    3.17s  4.60%   635ms    389KiB  1.02%  77.8KiB
 128×128×128  (CPU, Float64)        5    3.19s  4.62%   637ms    389KiB  1.02%  77.8KiB
 128×128×128  (GPU, Float32)        5   75.2ms  0.11%  15.0ms   4.29MiB  11.5%   880KiB
 128×128×128  (GPU, Float64)        5   75.1ms  0.11%  15.0ms   4.30MiB  11.5%   880KiB
 256×256×256  (CPU, Float32)        5    31.5s  45.7%   6.30s    389KiB  1.02%  77.8KiB
 256×256×256  (CPU, Float64)        5    29.2s  42.3%   5.83s    389KiB  1.02%  77.8KiB
 256×256×256  (GPU, Float32)        5    391ms  0.57%  78.1ms   4.59MiB  12.3%   940KiB
 256×256×256  (GPU, Float64)        5    368ms  0.53%  73.6ms   4.59MiB  12.3%   940KiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.072
 64× 64× 64 : 1.003
128×128×128 : 1.004
256×256×256 : 0.926

GPU Float64 -> Float32 speedup:
 32× 32× 32 : 0.970
 64× 64× 64 : 0.987
128×128×128 : 0.999
256×256×256 : 0.943

CPU -> GPU speedup:
 32× 32× 32  (Float32): 1.364
 32× 32× 32  (Float64): 1.508
 64× 64× 64  (Float32): 8.449
 64× 64× 64  (Float64): 8.588
128×128×128  (Float32): 42.209
128×128×128  (Float64): 42.411
256×256×256  (Float32): 80.638
256×256×256  (Float64): 79.211
```

## Tracers

This benchmark tests the performance impacts of running with various amounts of active
and passive tracers.

```
Julia Version 1.3.0
Commit 46ce4d7933 (2019-11-26 06:09 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, broadwell)
  GPU: Tesla V100-PCIE-32GB

 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
                      Tracer benchmarks                             Time                   Allocations      
                                                            ──────────────────────   ───────────────────────
                      Tot / % measured:                          37.6s / 9.69%           7.64GiB / 1.12%    

 Section                                            ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32 0 active +  0 passive (CPU, Float64)       10   60.0ms  1.65%  6.00ms    574KiB  0.64%  57.4KiB
  32× 32× 32 0 active +  1 passive (CPU, Float64)       10   68.4ms  1.88%  6.84ms    667KiB  0.74%  66.7KiB
  32× 32× 32 0 active +  2 passive (CPU, Float64)       10   76.8ms  2.11%  7.68ms    768KiB  0.85%  76.8KiB
  32× 32× 32 1 active +  0 passive (CPU, Float64)       10   69.2ms  1.90%  6.92ms    667KiB  0.74%  66.7KiB
  32× 32× 32 2 active +  0 passive (CPU, Float64)       10   78.7ms  2.16%  7.87ms    768KiB  0.85%  76.8KiB
  32× 32× 32 2 active +  3 passive (CPU, Float64)       10    104ms  2.86%  10.4ms   1.03MiB  1.17%   106KiB
  32× 32× 32 2 active +  5 passive (CPU, Float64)       10    123ms  3.38%  12.3ms   1.22MiB  1.39%   125KiB
  32× 32× 32 2 active + 10 passive (CPU, Float64)       10    177ms  4.86%  17.7ms   1.69MiB  1.92%   173KiB
 256×256×256 0 active +  0 passive (GPU, Float64)       10    237ms  6.50%  23.7ms   5.43MiB  6.17%   556KiB
 256×256×256 0 active +  1 passive (GPU, Float64)       10    266ms  7.29%  26.6ms   6.62MiB  7.52%   678KiB
 256×256×256 0 active +  2 passive (GPU, Float64)       10    297ms  8.16%  29.7ms   7.83MiB  8.89%   801KiB
 256×256×256 1 active +  0 passive (GPU, Float64)       10    268ms  7.35%  26.8ms   6.62MiB  7.52%   678KiB
 256×256×256 2 active +  0 passive (GPU, Float64)       10    303ms  8.32%  30.3ms   7.84MiB  8.91%   803KiB
 256×256×256 2 active +  3 passive (GPU, Float64)       10    403ms  11.1%  40.3ms   11.5MiB  13.1%  1.15MiB
 256×256×256 2 active +  5 passive (GPU, Float64)       10    472ms  13.0%  47.2ms   14.1MiB  16.0%  1.41MiB
 256×256×256 2 active + 10 passive (GPU, Float64)       10    641ms  17.6%  64.1ms   20.8MiB  23.6%  2.08MiB
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Turbulence closures

This benchmark tests the performance impacts of various turbulence closures and large eddy
simulation (LES) models.

```
Julia Version 1.3.0
Commit 46ce4d7933 (2019-11-26 06:09 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2603 v4 @ 1.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-6.0.1 (ORCJIT, broadwell)
  GPU: Tesla V100-PCIE-32GB

 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                    Turbulence closure benchmarks                            Time                   Allocations      
                                                                     ──────────────────────   ───────────────────────
                          Tot / % measured:                               31.0s / 78.5%           1.31GiB / 3.92%    

 Section                                                     ncalls     time   %tot     avg     alloc   %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32 AnisotropicDiffusivity (CPU, Float64)       10   78.1ms  0.32%  7.81ms    769KiB  1.42%  76.9KiB
  32× 32× 32 AnisotropicDiffusivity (GPU, Float64)       10   43.0ms  0.18%  4.30ms   7.86MiB  14.9%   805KiB
  32× 32× 32 IsotropicDiffusivity (CPU, Float64)         10   78.7ms  0.32%  7.87ms    768KiB  1.42%  76.8KiB
  32× 32× 32 IsotropicDiffusivity (GPU, Float64)         10   44.5ms  0.18%  4.45ms   7.84MiB  14.9%   803KiB
  32× 32× 32 SmagorinskyLilly (CPU, Float64)                     10    189ms  0.78%  18.9ms    778KiB  1.44%  77.8KiB
  32× 32× 32 SmagorinskyLilly (GPU, Float64)                     10   45.7ms  0.19%  4.57ms   8.43MiB  16.0%   863KiB
 128×128×128 AnisotropicDiffusivity (CPU, Float64)       10    5.54s  22.8%   554ms    769KiB  1.42%  76.9KiB
 128×128×128 AnisotropicDiffusivity (GPU, Float64)       10   53.5ms  0.22%  5.35ms   7.86MiB  14.9%   805KiB
 128×128×128 IsotropicDiffusivity (CPU, Float64)         10    5.53s  22.7%   553ms    768KiB  1.42%  76.8KiB
 128×128×128 IsotropicDiffusivity (GPU, Float64)         10   54.1ms  0.22%  5.41ms   7.84MiB  14.9%   803KiB
 128×128×128 SmagorinskyLilly (CPU, Float64)                     10    12.6s  51.8%   1.26s    778KiB  1.44%  77.8KiB
 128×128×128 SmagorinskyLilly (GPU, Float64)                     10   75.6ms  0.31%  7.56ms   8.43MiB  16.0%   863KiB
 ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```
