
# Performance benchmarks

The performance benchmarking scripts in the
[`benchmarks`](https://github.com/CliMA/Oceananigans.jl/tree/master/benchmark)
directory of the git repository can be run to benchmark Oceananigans.jl on your machine.
They use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) to collect data and [PrettyTables.jl](https://github.com/ronisbr/PrettyTables.jl)  to nicely
format the benchmark results.

## Static ocean

This is a benchmark of a simple "static ocean" configuration. The time stepping and Poisson
solver still takes the same amount of time whether the ocean is static or active, so it is
indicative of actual performance. It tests the performance of a bare-bones
horizontally-periodic model with `topology = (Periodic, Periodic, Bounded)`.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ──────────────────────────────────────────────────────────────────────────────────────
        Static ocean benchmarks                Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 291s / 29.6%           27.7GiB / 0.50%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  16× 16× 16  [CPU, Float32]       10   15.6ms  0.02%  1.56ms   2.61MiB  1.84%   267KiB
  16× 16× 16  [CPU, Float64]       10   16.9ms  0.02%  1.69ms   2.61MiB  1.84%   267KiB
  16× 16× 16  [GPU, Float32]       10   53.4ms  0.06%  5.34ms   11.5MiB  8.14%  1.15MiB
  16× 16× 16  [GPU, Float64]       10   69.7ms  0.08%  6.97ms   11.5MiB  8.14%  1.15MiB
  32× 32× 32  [CPU, Float32]       10   54.6ms  0.06%  5.46ms   2.61MiB  1.84%   267KiB
  32× 32× 32  [CPU, Float64]       10   57.1ms  0.07%  5.71ms   2.61MiB  1.84%   267KiB
  32× 32× 32  [GPU, Float32]       10   57.5ms  0.07%  5.75ms   11.6MiB  8.15%  1.16MiB
  32× 32× 32  [GPU, Float64]       10   75.0ms  0.09%  7.50ms   11.6MiB  8.16%  1.16MiB
  64× 64× 64  [CPU, Float32]       10    424ms  0.49%  42.4ms   2.61MiB  1.84%   267KiB
  64× 64× 64  [CPU, Float64]       10    425ms  0.49%  42.5ms   2.61MiB  1.84%   267KiB
  64× 64× 64  [GPU, Float32]       10   61.7ms  0.07%  6.17ms   11.6MiB  8.16%  1.16MiB
  64× 64× 64  [GPU, Float64]       10   82.4ms  0.10%  8.24ms   11.6MiB  8.17%  1.16MiB
 128×128×128  [CPU, Float32]       10    3.67s  4.26%   367ms   2.61MiB  1.84%   267KiB
 128×128×128  [CPU, Float64]       10    3.64s  4.23%   364ms   2.61MiB  1.84%   267KiB
 128×128×128  [GPU, Float32]       10   74.8ms  0.09%  7.48ms   11.6MiB  8.16%  1.16MiB
 128×128×128  [GPU, Float64]       10   94.0ms  0.11%  9.40ms   11.6MiB  8.17%  1.16MiB
 256×256×256  [CPU, Float32]       10    38.5s  44.8%   3.85s   2.61MiB  1.84%   267KiB
 256×256×256  [CPU, Float64]       10    37.9s  44.1%   3.79s   2.61MiB  1.84%   267KiB
 256×256×256  [GPU, Float32]       10    350ms  0.41%  35.0ms   11.6MiB  8.18%  1.16MiB
 256×256×256  [GPU, Float64]       10    352ms  0.41%  35.2ms   11.6MiB  8.17%  1.16MiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 16× 16× 16 : 1.084
 32× 32× 32 : 1.046
 64× 64× 64 : 1.000
128×128×128 : 0.993
256×256×256 : 0.986

GPU Float64 -> Float32 speedup:
 16× 16× 16 : 1.304
 32× 32× 32 : 1.303
 64× 64× 64 : 1.335
128×128×128 : 1.257
256×256×256 : 1.004

CPU -> GPU speedup:
 16× 16× 16  [Float32]: 0.291
 16× 16× 16  [Float64]: 0.242
 32× 32× 32  [Float32]: 0.949
 32× 32× 32  [Float64]: 0.762
 64× 64× 64  [Float32]: 6.876
 64× 64× 64  [Float64]: 5.152
128×128×128  [Float32]: 49.036
128×128×128  [Float64]: 38.730
256×256×256  [Float32]: 109.868
256×256×256  [Float64]: 107.863
```

## Channel

This benchmark tests the channel model (`topology = (Periodic, Bounded, Bounded)`)
configuration which can be slower due to the use of a more complicated algorithm
(involving 2D cosine transforms) for the pressure solver in the current version
of Oceananigans.

```
Oceananigans v0.34.0 (DEVELOPMENT BRANCH)
Julia Version 1.4.2
Commit 44fa15b150* (2020-05-23 18:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-8.0.1 (ORCJIT, skylake)
  GPU: TITAN V

 ──────────────────────────────────────────────────────────────────────────────────────
           Channel benchmarks                  Time                   Allocations      
                                       ──────────────────────   ───────────────────────
           Tot / % measured:                 453s / 19.5%           26.3GiB / 0.48%    

 Section                       ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────
  32× 32× 32  [CPU, Float32]       10   58.5ms  0.07%  5.85ms   2.84MiB  2.22%   291KiB
  32× 32× 32  [CPU, Float64]       10   60.8ms  0.07%  6.08ms   2.85MiB  2.22%   291KiB
  32× 32× 32  [GPU, Float32]       10   68.7ms  0.08%  6.87ms   12.6MiB  9.85%  1.26MiB
  32× 32× 32  [GPU, Float64]       10   88.2ms  0.10%  8.82ms   12.6MiB  9.85%  1.26MiB
  64× 64× 64  [CPU, Float32]       10    459ms  0.52%  45.9ms   2.84MiB  2.22%   291KiB
  64× 64× 64  [CPU, Float64]       10    442ms  0.50%  44.2ms   2.85MiB  2.22%   291KiB
  64× 64× 64  [GPU, Float32]       10   91.0ms  0.10%  9.10ms   12.8MiB  10.0%  1.28MiB
  64× 64× 64  [GPU, Float64]       10    108ms  0.12%  10.8ms   12.8MiB  10.0%  1.28MiB
 128×128×128  [CPU, Float32]       10    3.87s  4.38%   387ms   2.84MiB  2.22%   291KiB
 128×128×128  [CPU, Float64]       10    3.92s  4.44%   392ms   2.85MiB  2.22%   291KiB
 128×128×128  [GPU, Float32]       10    145ms  0.16%  14.5ms   13.2MiB  10.3%  1.32MiB
 128×128×128  [GPU, Float64]       10    163ms  0.18%  16.3ms   13.2MiB  10.3%  1.32MiB
 256×256×256  [CPU, Float32]       10    38.6s  43.6%   3.86s   2.85MiB  2.22%   292KiB
 256×256×256  [CPU, Float64]       10    38.7s  43.8%   3.87s   2.85MiB  2.22%   292KiB
 256×256×256  [GPU, Float32]       10    805ms  0.91%  80.5ms   14.0MiB  10.9%  1.40MiB
 256×256×256  [GPU, Float64]       10    805ms  0.91%  80.5ms   14.0MiB  10.9%  1.40MiB
 ──────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.040
 64× 64× 64 : 0.963
128×128×128 : 1.015
256×256×256 : 1.004

GPU Float64 -> Float32 speedup:
 32× 32× 32 : 1.283
 64× 64× 64 : 1.188
128×128×128 : 1.120
256×256×256 : 0.999

CPU -> GPU speedup:
 32× 32× 32  [Float32]: 0.851
 32× 32× 32  [Float64]: 0.689
 64× 64× 64  [Float32]: 5.044
 64× 64× 64  [Float64]: 4.088
128×128×128  [Float32]: 26.602
128×128×128  [Float64]: 24.097
256×256×256  [Float32]: 47.891
256×256×256  [Float64]: 48.116
```

## Tracers

This benchmark tests the performance impacts of running with various amounts of active
and passive tracers and compares the difference in speedup going from CPU to GPU. Number of tracers are listed in the tracers column as such: (active, passive). 

```

Oceananigans v0.58.1
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cascadelake)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :
  GPU: Tesla V100-SXM2-32GB

                                       Arbitrary tracers benchmarks
┌───────────────┬─────────┬───────────┬───────────┬───────────┬───────────┬────────────┬────────┬─────────┐
│ Architectures │ tracers │       min │    median │      mean │       max │     memory │ allocs │ samples │
├───────────────┼─────────┼───────────┼───────────┼───────────┼───────────┼────────────┼────────┼─────────┤
│           CPU │  (0, 0) │   1.439 s │   1.440 s │   1.440 s │   1.441 s │ 908.03 KiB │   1656 │       4 │
│           CPU │  (0, 1) │   1.539 s │   1.574 s │   1.575 s │   1.613 s │   1.24 MiB │   1942 │       4 │
│           CPU │  (0, 2) │   1.668 s │   1.669 s │   1.670 s │   1.671 s │   1.76 MiB │   2291 │       3 │
│           CPU │  (1, 0) │   1.527 s │   1.532 s │   1.532 s │   1.536 s │   1.24 MiB │   1942 │       4 │
│           CPU │  (2, 0) │   1.690 s │   1.697 s │   1.695 s │   1.698 s │   1.77 MiB │   2301 │       3 │
│           CPU │  (2, 3) │   2.234 s │   2.239 s │   2.241 s │   2.251 s │   3.59 MiB │   3928 │       3 │
│           CPU │  (2, 5) │   2.755 s │   2.838 s │   2.838 s │   2.921 s │   5.18 MiB │   4908 │       2 │
│           CPU │ (2, 10) │   3.588 s │   3.748 s │   3.748 s │   3.908 s │  10.39 MiB │   7682 │       2 │
│           GPU │  (0, 0) │  9.702 ms │ 12.755 ms │ 12.458 ms │ 12.894 ms │   1.59 MiB │  12321 │      10 │
│           GPU │  (0, 1) │ 13.863 ms │ 13.956 ms │ 14.184 ms │ 16.297 ms │   2.20 MiB │  14294 │      10 │
│           GPU │  (0, 2) │ 15.166 ms │ 15.230 ms │ 15.700 ms │ 19.893 ms │   2.93 MiB │  15967 │      10 │
│           GPU │  (1, 0) │ 13.740 ms │ 13.838 ms │ 14.740 ms │ 22.940 ms │   2.20 MiB │  14278 │      10 │
│           GPU │  (2, 0) │ 15.103 ms │ 15.199 ms │ 16.265 ms │ 25.906 ms │   2.93 MiB │  15913 │      10 │
│           GPU │  (2, 3) │ 13.981 ms │ 18.856 ms │ 18.520 ms │ 20.519 ms │   5.56 MiB │  17974 │      10 │
│           GPU │  (2, 5) │ 15.824 ms │ 21.211 ms │ 21.064 ms │ 24.897 ms │   7.86 MiB │  23938 │      10 │
│           GPU │ (2, 10) │ 22.085 ms │ 27.236 ms │ 28.231 ms │ 38.295 ms │  15.02 MiB │  31086 │      10 │
└───────────────┴─────────┴───────────┴───────────┴───────────┴───────────┴────────────┴────────┴─────────┘

  Arbitrary tracers CPU to GPU speedup
┌─────────┬─────────┬─────────┬─────────┐
│ tracers │ speedup │  memory │  allocs │
├─────────┼─────────┼─────────┼─────────┤
│  (0, 0) │ 112.881 │ 1.78792 │ 7.44022 │
│  (0, 1) │ 112.761 │ 1.77743 │ 7.36045 │
│  (0, 2) │ 109.618 │  1.6627 │ 6.96945 │
│  (1, 0) │ 110.717 │ 1.77723 │ 7.35221 │
│  (2, 0) │ 111.678 │ 1.66267 │ 6.91569 │
│  (2, 3) │ 118.737 │ 1.55043 │ 4.57587 │
│  (2, 5) │ 133.803 │  1.5155 │ 4.87734 │
│ (2, 10) │ 137.615 │ 1.44535 │  4.0466 │
└─────────┴─────────┴─────────┴─────────┘

       Arbitrary tracers relative performance (CPU)
┌───────────────┬─────────┬──────────┬─────────┬─────────┐
│ Architectures │ tracers │ slowdown │  memory │  allocs │
├───────────────┼─────────┼──────────┼─────────┼─────────┤
│           CPU │  (0, 0) │      1.0 │     1.0 │     1.0 │
│           CPU │  (0, 1) │  1.09293 │ 1.39873 │ 1.17271 │
│           CPU │  (0, 2) │  1.15948 │ 1.99019 │ 1.38345 │
│           CPU │  (1, 0) │  1.06409 │ 1.39873 │ 1.17271 │
│           CPU │  (2, 0) │  1.17887 │ 1.99054 │ 1.38949 │
│           CPU │  (2, 3) │  1.55493 │ 4.04677 │ 2.37198 │
│           CPU │  (2, 5) │  1.97115 │ 5.84537 │ 2.96377 │
│           CPU │ (2, 10) │   2.6031 │ 11.7179 │ 4.63889 │
└───────────────┴─────────┴──────────┴─────────┴─────────┘

       Arbitrary tracers relative performance (GPU)
┌───────────────┬─────────┬──────────┬─────────┬─────────┐
│ Architectures │ tracers │ slowdown │  memory │  allocs │
├───────────────┼─────────┼──────────┼─────────┼─────────┤
│           GPU │  (0, 0) │      1.0 │     1.0 │     1.0 │
│           GPU │  (0, 1) │   1.0941 │ 1.39053 │ 1.16013 │
│           GPU │  (0, 2) │  1.19399 │ 1.85081 │ 1.29592 │
│           GPU │  (1, 0) │  1.08489 │ 1.39037 │ 1.15883 │
│           GPU │  (2, 0) │  1.19157 │ 1.85109 │ 1.29153 │
│           GPU │  (2, 3) │  1.47824 │ 3.50924 │ 1.45881 │
│           GPU │  (2, 5) │  1.66293 │ 4.95474 │ 1.94286 │
│           GPU │ (2, 10) │  2.13524 │ 9.47276 │ 2.52301 │
└───────────────┴─────────┴──────────┴─────────┴─────────┘
```

## Turbulence closures

This benchmark tests the performance impacts of various turbulent diffusivity closures
and large eddy simulation (LES) models as well as how much speedup they experience going from CPU to GPU.

```
Oceananigans v0.58.1
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cascadelake)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :
  GPU: Tesla V100-SXM2-32GB

                                                  Turbulence closure benchmarks
┌───────────────┬──────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬──────────┬────────┬─────────┐
│ Architectures │                         Closures │       min │    median │      mean │       max │   memory │ allocs │ samples │
├───────────────┼──────────────────────────────────┼───────────┼───────────┼───────────┼───────────┼──────────┼────────┼─────────┤
│           CPU │ AnisotropicBiharmonicDiffusivity │   3.634 s │   3.637 s │   3.637 s │   3.639 s │ 1.77 MiB │   2316 │       2 │
│           CPU │           AnisotropicDiffusivity │   2.045 s │   2.052 s │   2.059 s │   2.079 s │ 1.77 MiB │   2316 │       3 │
│           CPU │    AnisotropicMinimumDissipation │   3.240 s │   3.240 s │   3.240 s │   3.241 s │ 2.09 MiB │   2763 │       2 │
│           CPU │             IsotropicDiffusivity │   2.342 s │   2.344 s │   2.344 s │   2.345 s │ 1.77 MiB │   2316 │       3 │
│           CPU │                 SmagorinskyLilly │   3.501 s │   3.504 s │   3.504 s │   3.507 s │ 2.03 MiB │   2486 │       2 │
│           CPU │              TwoDimensionalLeith │   4.813 s │   4.820 s │   4.820 s │   4.828 s │ 1.88 MiB │   2481 │       2 │
│           GPU │ AnisotropicBiharmonicDiffusivity │ 24.699 ms │ 24.837 ms │ 26.946 ms │ 46.029 ms │ 3.16 MiB │  29911 │      10 │
│           GPU │           AnisotropicDiffusivity │ 16.115 ms │ 16.184 ms │ 16.454 ms │ 18.978 ms │ 2.97 MiB │  17169 │      10 │
│           GPU │    AnisotropicMinimumDissipation │ 15.858 ms │ 25.856 ms │ 24.874 ms │ 26.014 ms │ 3.57 MiB │  24574 │      10 │
│           GPU │             IsotropicDiffusivity │ 14.442 ms │ 17.415 ms │ 17.134 ms │ 17.513 ms │ 2.99 MiB │  19135 │      10 │
│           GPU │                 SmagorinskyLilly │ 16.315 ms │ 23.969 ms │ 23.213 ms │ 24.059 ms │ 3.86 MiB │  24514 │      10 │
│           GPU │              TwoDimensionalLeith │ 34.470 ms │ 34.628 ms │ 35.535 ms │ 43.798 ms │ 3.56 MiB │  45291 │      10 │
└───────────────┴──────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴──────────┴────────┴─────────┘

              Turbulence closure CPU to GPU speedup
┌──────────────────────────────────┬─────────┬─────────┬─────────┐
│                         Closures │ speedup │  memory │  allocs │
├──────────────────────────────────┼─────────┼─────────┼─────────┤
│ AnisotropicBiharmonicDiffusivity │ 146.428 │ 1.78781 │ 12.9149 │
│           AnisotropicDiffusivity │ 126.804 │ 1.67787 │ 7.41321 │
│    AnisotropicMinimumDissipation │ 125.324 │ 1.70856 │ 8.89396 │
│             IsotropicDiffusivity │ 134.607 │ 1.69269 │ 8.26209 │
│                 SmagorinskyLilly │ 146.187 │ 1.89602 │ 9.86082 │
│              TwoDimensionalLeith │ 139.196 │ 1.89218 │ 18.2551 │
└──────────────────────────────────┴─────────┴─────────┴─────────┘

```
## Shallow Water Model

This benchmark tests the performance of the shallow water model run in a doubly periodic domain (`topology = (Periodic, Periodic, Flat)`)
on a CPU versus a GPU.  We find that with the `WENO5` advection scheme we get a maximum speedup of more than 400 times on a '16384^2' grid.
```
Oceananigans v0.58.1
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cascadelake)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :
  GPU: Tesla V100-SXM2-32GB

                                              Shallow water model benchmarks
┌───────────────┬─────────────┬───────┬────────────┬────────────┬────────────┬────────────┬───────────┬────────┬─────────┐
│ Architectures │ Float_types │    Ns │        min │     median │       mean │        max │    memory │ allocs │ samples │
├───────────────┼─────────────┼───────┼────────────┼────────────┼────────────┼────────────┼───────────┼────────┼─────────┤
│           CPU │     Float64 │    32 │   2.677 ms │   2.876 ms │   3.047 ms │   4.806 ms │  1.36 MiB │   2253 │      10 │
│           CPU │     Float64 │    64 │   5.795 ms │   5.890 ms │   6.073 ms │   7.770 ms │  1.36 MiB │   2255 │      10 │
│           CPU │     Float64 │   128 │  16.979 ms │  17.350 ms │  17.578 ms │  19.993 ms │  1.36 MiB │   2255 │      10 │
│           CPU │     Float64 │   256 │  62.543 ms │  63.222 ms │  63.544 ms │  67.347 ms │  1.36 MiB │   2255 │      10 │
│           CPU │     Float64 │   512 │ 250.149 ms │ 251.023 ms │ 251.092 ms │ 252.389 ms │  1.36 MiB │   2315 │      10 │
│           CPU │     Float64 │  1024 │ 990.901 ms │ 993.115 ms │ 993.360 ms │ 996.091 ms │  1.36 MiB │   2315 │       6 │
│           CPU │     Float64 │  2048 │    4.002 s │    4.004 s │    4.004 s │    4.007 s │  1.36 MiB │   2315 │       2 │
│           CPU │     Float64 │  4096 │   16.371 s │   16.371 s │   16.371 s │   16.371 s │  1.36 MiB │   2315 │       1 │
│           CPU │     Float64 │  8192 │   64.657 s │   64.657 s │   64.657 s │   64.657 s │  1.36 MiB │   2315 │       1 │
│           CPU │     Float64 │ 16384 │  290.423 s │  290.423 s │  290.423 s │  290.423 s │  1.36 MiB │   2315 │       1 │
│           GPU │     Float64 │    32 │   3.468 ms │   3.656 ms │   3.745 ms │   4.695 ms │  1.82 MiB │   5687 │      10 │
│           GPU │     Float64 │    64 │   3.722 ms │   3.903 ms │   4.050 ms │   5.671 ms │  1.82 MiB │   5687 │      10 │
│           GPU │     Float64 │   128 │   3.519 ms │   3.808 ms │   4.042 ms │   6.372 ms │  1.82 MiB │   5687 │      10 │
│           GPU │     Float64 │   256 │   3.822 ms │   4.153 ms │   4.288 ms │   5.810 ms │  1.82 MiB │   5687 │      10 │
│           GPU │     Float64 │   512 │   4.637 ms │   4.932 ms │   4.961 ms │   5.728 ms │  1.82 MiB │   5765 │      10 │
│           GPU │     Float64 │  1024 │   3.240 ms │   3.424 ms │   3.527 ms │   4.553 ms │  1.82 MiB │   5799 │      10 │
│           GPU │     Float64 │  2048 │  10.783 ms │  10.800 ms │  11.498 ms │  17.824 ms │  1.98 MiB │  16305 │      10 │
│           GPU │     Float64 │  4096 │  41.880 ms │  41.911 ms │  42.485 ms │  47.627 ms │  2.67 MiB │  61033 │      10 │
│           GPU │     Float64 │  8192 │ 166.751 ms │ 166.800 ms │ 166.847 ms │ 167.129 ms │  5.21 MiB │ 227593 │      10 │
│           GPU │     Float64 │ 16384 │ 681.129 ms │ 681.249 ms │ 681.301 ms │ 681.583 ms │ 16.59 MiB │ 973627 │       8 │
└───────────────┴─────────────┴───────┴────────────┴────────────┴────────────┴────────────┴───────────┴────────┴─────────┘

        Shallow water model CPU to GPU speedup
┌─────────────┬───────┬──────────┬─────────┬─────────┐
│ Float_types │    Ns │  speedup │  memory │  allocs │
├─────────────┼───────┼──────────┼─────────┼─────────┤
│     Float64 │    32 │ 0.786715 │ 1.33777 │ 2.52419 │
│     Float64 │    64 │  1.50931 │ 1.33774 │ 2.52195 │
│     Float64 │   128 │  4.55587 │ 1.33774 │ 2.52195 │
│     Float64 │   256 │  15.2238 │ 1.33774 │ 2.52195 │
│     Float64 │   512 │  50.8995 │ 1.33771 │ 2.49028 │
│     Float64 │  1024 │  290.085 │ 1.33809 │ 2.50497 │
│     Float64 │  2048 │  370.777 │ 1.45575 │  7.0432 │
│     Float64 │  4096 │  390.617 │ 1.95667 │ 26.3641 │
│     Float64 │  8192 │  387.632 │ 3.82201 │ 98.3123 │
│     Float64 │ 16384 │   426.31 │  12.177 │ 420.573 │
└─────────────┴───────┴──────────┴─────────┴─────────┘
```
As shown in the graph below, speedups increase sharply starting at grid size `512^2` and then plateau off around 400 times at grid size `4096^2` and beyond.

![shallow_water_speedup](https://user-images.githubusercontent.com/45054739/128793049-7bcbabaa-2d66-4209-a311-b02729fb93fa.png)

The time graph below shows that times on GPU are negligebly small up until grid size `1024^2` where it starts to scale similarly to times on CPU.

![shallow_water_times](https://user-images.githubusercontent.com/45054739/128793311-e4bbfd5a-aea8-4cdc-bee8-cb71128ff5fe.png)


