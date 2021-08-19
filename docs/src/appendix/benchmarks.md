
# Performance benchmarks

The performance benchmarking scripts in the
[`benchmarks`](https://github.com/CliMA/Oceananigans.jl/tree/master/benchmark)
directory of the git repository can be run to benchmark Oceananigans.jl on your machine.
They use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) to collect data and [PrettyTables.jl](https://github.com/ronisbr/PrettyTables.jl)  to nicely
format the benchmark results.


## Shallow Water Model

This benchmark tests the performance of the shallow water model run in a doubly periodic domain (`topology = (Periodic, Periodic, Flat)`)
on a CPU versus a GPU.  We find that with the `WENO5` advection scheme we get a maximum speedup of more than 400 times on a `16384^2` grid.
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
As shown in the graph below, speedups increase sharply starting at grid size `512^2` and then plateau off at around 400 times at grid size `4096^2` and beyond.

![shallow_water_speedup](https://user-images.githubusercontent.com/45054739/128793049-7bcbabaa-2d66-4209-a311-b02729fb93fa.png)

The time graph below shows that execution times on GPU are negligebly small up until grid size `1024^2` where it starts to scale similarly to times on CPU.

![shallow_water_times](https://user-images.githubusercontent.com/45054739/128793311-e4bbfd5a-aea8-4cdc-bee8-cb71128ff5fe.png)

## Nonhydrostatic Model

Similar to to shallow water model, the nonhydrostatic model benchmark tests for its performance on both a CPU and a GPU. It was also benchmarked with the `WENO5` advection scheme. The nonhydrostatic model is 3-dimensional unlike the 2-dimensional shallow water model. Total number of grid points is Ns cubed.
```
Oceananigans v0.58.8
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, cascadelake)
Environment:
  EBVERSIONJULIA = 1.6.1
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1/easybuild/avx2-Core-julia-1.6.1-easybuild-devel
  JULIA_LOAD_PATH = :
  GPU: Tesla V100-SXM2-32GB

                                            Nonhydrostatic model benchmarks
┌───────────────┬─────────────┬─────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│ Architectures │ Float_types │  Ns │        min │     median │       mean │        max │   memory │ allocs │ samples │
├───────────────┼─────────────┼─────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│           CPU │     Float32 │  32 │  34.822 ms │  34.872 ms │  35.278 ms │  38.143 ms │ 1.38 MiB │   2302 │      10 │
│           CPU │     Float32 │  64 │ 265.408 ms │ 265.571 ms │ 265.768 ms │ 267.765 ms │ 1.38 MiB │   2302 │      10 │
│           CPU │     Float32 │ 128 │    2.135 s │    2.135 s │    2.136 s │    2.138 s │ 1.38 MiB │   2302 │       3 │
│           CPU │     Float32 │ 256 │   17.405 s │   17.405 s │   17.405 s │   17.405 s │ 1.38 MiB │   2302 │       1 │
│           CPU │     Float64 │  32 │  37.022 ms │  37.179 ms │  37.335 ms │  39.017 ms │ 1.77 MiB │   2302 │      10 │
│           CPU │     Float64 │  64 │ 287.944 ms │ 288.154 ms │ 288.469 ms │ 290.838 ms │ 1.77 MiB │   2302 │      10 │
│           CPU │     Float64 │ 128 │    2.326 s │    2.326 s │    2.326 s │    2.327 s │ 1.77 MiB │   2302 │       3 │
│           CPU │     Float64 │ 256 │   19.561 s │   19.561 s │   19.561 s │   19.561 s │ 1.77 MiB │   2302 │       1 │
│           GPU │     Float32 │  32 │   4.154 ms │   4.250 ms │   4.361 ms │   5.557 ms │ 2.13 MiB │   6033 │      10 │
│           GPU │     Float32 │  64 │   3.383 ms │   3.425 ms │   3.889 ms │   8.028 ms │ 2.13 MiB │   6077 │      10 │
│           GPU │     Float32 │ 128 │   5.564 ms │   5.580 ms │   6.095 ms │  10.725 ms │ 2.15 MiB │   7477 │      10 │
│           GPU │     Float32 │ 256 │  38.685 ms │  38.797 ms │  39.548 ms │  46.442 ms │ 2.46 MiB │  27721 │      10 │
│           GPU │     Float64 │  32 │   3.309 ms │   3.634 ms │   3.802 ms │   5.844 ms │ 2.68 MiB │   6033 │      10 │
│           GPU │     Float64 │  64 │   3.330 ms │   3.648 ms │   4.008 ms │   7.808 ms │ 2.68 MiB │   6071 │      10 │
│           GPU │     Float64 │ 128 │   7.209 ms │   7.323 ms │   8.313 ms │  17.259 ms │ 2.71 MiB │   8515 │      10 │
│           GPU │     Float64 │ 256 │  46.614 ms │  56.444 ms │  55.461 ms │  56.563 ms │ 3.17 MiB │  38253 │      10 │
└───────────────┴─────────────┴─────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

      Nonhydrostatic model CPU to GPU speedup
┌─────────────┬─────┬─────────┬─────────┬─────────┐
│ Float_types │  Ns │ speedup │  memory │  allocs │
├─────────────┼─────┼─────────┼─────────┼─────────┤
│     Float32 │  32 │ 8.20434 │ 1.53786 │ 2.62076 │
│     Float32 │  64 │ 77.5308 │ 1.53835 │ 2.63988 │
│     Float32 │ 128 │ 382.591 │ 1.55378 │ 3.24805 │
│     Float32 │ 256 │ 448.619 │ 1.77688 │ 12.0421 │
│     Float64 │  32 │ 10.2308 │ 1.51613 │ 2.62076 │
│     Float64 │  64 │ 78.9952 │ 1.51646 │ 2.63727 │
│     Float64 │ 128 │ 317.663 │ 1.53759 │ 3.69896 │
│     Float64 │ 256 │ 346.554 │ 1.79466 │ 16.6173 │
└─────────────┴─────┴─────────┴─────────┴─────────┘
```

Like the shallow water model, it can be seen at grid size `64^3` that the GPU is beginning to be saturated as speedups rapidly increase. At grid sizes `128^3` and `256^3` we see the speedup stablise to around 400 times.

![incompressible_speedup](https://user-images.githubusercontent.com/45054739/129825248-adb8dfe5-e9ea-4321-bd11-fb415d81e2cb.png)

For both float types, the benchmarked GPU times of the nonhydrostatic model starts to scale like its CPU times when grid size reaches `128^3`.

![incompressible_times](https://user-images.githubusercontent.com/45054739/129825253-0d5739d9-f0a7-476e-8152-4ee462b71ad5.png)

## Distributed Shallow Water Model

By using `MPI.jl` the shallow water model can be run on multiple CPUs and multiple GPUs. For the benchmark results shown below, each rank is run on one CPU core and each uses a distinct GPU if applicable. 

### Weak Scaling Shallow Water Model
```
Oceananigans v0.58.2
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :

                                  Shallow water model weak scaling benchmark
┌───────────────┬──────────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│          size │    ranks │        min │     median │       mean │        max │   memory │ allocs │ samples │
├───────────────┼──────────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│   (4096, 256) │   (1, 1) │ 363.885 ms │ 364.185 ms │ 364.911 ms │ 370.414 ms │ 1.60 MiB │   2774 │      10 │
│   (4096, 512) │   (1, 2) │ 370.782 ms │ 375.032 ms │ 375.801 ms │ 394.781 ms │ 1.49 MiB │   3116 │      20 │
│  (4096, 1024) │   (1, 4) │ 369.648 ms │ 369.973 ms │ 371.613 ms │ 399.526 ms │ 1.49 MiB │   3116 │      40 │
│  (4096, 2048) │   (1, 8) │ 377.386 ms │ 379.982 ms │ 382.732 ms │ 432.787 ms │ 1.49 MiB │   3116 │      80 │
│  (4096, 4096) │  (1, 16) │ 388.336 ms │ 395.473 ms │ 400.079 ms │ 496.598 ms │ 1.49 MiB │   3116 │     160 │
│  (4096, 8192) │  (1, 32) │ 403.565 ms │ 447.136 ms │ 449.138 ms │ 545.945 ms │ 1.49 MiB │   3116 │     320 │
│ (4096, 16384) │  (1, 64) │ 397.965 ms │ 441.627 ms │ 453.465 ms │ 619.493 ms │ 1.49 MiB │   3125 │     640 │
│ (4096, 32768) │ (1, 128) │ 400.481 ms │ 447.789 ms │ 448.692 ms │ 590.028 ms │ 1.49 MiB │   3125 │    1280 │
└───────────────┴──────────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

                Shallow water model weak scaling speedup
┌───────────────┬──────────┬──────────┬────────────┬──────────┬─────────┐
│          size │    ranks │ slowdown │ efficiency │   memory │  allocs │
├───────────────┼──────────┼──────────┼────────────┼──────────┼─────────┤
│   (4096, 256) │   (1, 1) │      1.0 │        1.0 │      1.0 │     1.0 │
│   (4096, 512) │   (1, 2) │  1.02978 │   0.971077 │ 0.930602 │ 1.12329 │
│  (4096, 1024) │   (1, 4) │  1.01589 │   0.984355 │ 0.930602 │ 1.12329 │
│  (4096, 2048) │   (1, 8) │  1.04338 │   0.958427 │ 0.930602 │ 1.12329 │
│  (4096, 4096) │  (1, 16) │  1.08591 │   0.920886 │ 0.930602 │ 1.12329 │
│  (4096, 8192) │  (1, 32) │  1.22777 │   0.814484 │ 0.930602 │ 1.12329 │
│ (4096, 16384) │  (1, 64) │  1.21264 │   0.824644 │ 0.930687 │ 1.12653 │
│ (4096, 32768) │ (1, 128) │  1.22957 │   0.813296 │ 0.930687 │ 1.12653 │
└───────────────┴──────────┴──────────┴────────────┴──────────┴─────────┘
```

As seen in the tables above and in the graph below, efficiency drops off to around 80% and remains as such from 16 to 128 ranks. GPUs are not used in this or the next benchmark setup. 

![ws_shallow_water_efficiency](https://user-images.githubusercontent.com/45054739/129826042-6ed4345b-b53a-49af-b375-6b7f11f53f31.png)

### Strong Scaling Shallow Water Model
```
Oceananigans v0.58.2
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :

                                Shallow water model strong scaling benchmark
┌──────────────┬──────────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│         size │    ranks │        min │     median │       mean │        max │   memory │ allocs │ samples │
├──────────────┼──────────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│ (4096, 4096) │   (1, 1) │    5.694 s │    5.694 s │    5.694 s │    5.694 s │ 1.60 MiB │   2804 │       1 │
│ (4096, 4096) │   (1, 2) │    2.865 s │    2.865 s │    2.866 s │    2.869 s │ 1.49 MiB │   3146 │       4 │
│ (4096, 4096) │   (1, 4) │    1.435 s │    1.437 s │    1.441 s │    1.475 s │ 1.49 MiB │   3146 │      16 │
│ (4096, 4096) │   (1, 8) │ 732.711 ms │ 736.394 ms │ 738.930 ms │ 776.773 ms │ 1.49 MiB │   3146 │      56 │
│ (4096, 4096) │  (1, 16) │ 389.211 ms │ 395.749 ms │ 396.813 ms │ 433.332 ms │ 1.49 MiB │   3116 │     160 │
│ (4096, 4096) │  (1, 32) │ 197.894 ms │ 219.211 ms │ 236.780 ms │ 367.188 ms │ 1.49 MiB │   3116 │     320 │
│ (4096, 4096) │  (1, 64) │ 101.520 ms │ 112.606 ms │ 116.809 ms │ 221.497 ms │ 1.49 MiB │   3125 │     640 │
│ (4096, 4096) │ (1, 128) │  51.452 ms │  60.256 ms │  70.959 ms │ 232.309 ms │ 1.49 MiB │   3125 │    1280 │
└──────────────┴──────────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

              Shallow water model strong scaling speedup
┌──────────────┬──────────┬─────────┬────────────┬──────────┬─────────┐
│         size │    ranks │ speedup │ efficiency │   memory │  allocs │
├──────────────┼──────────┼─────────┼────────────┼──────────┼─────────┤
│ (4096, 4096) │   (1, 1) │     1.0 │        1.0 │      1.0 │     1.0 │
│ (4096, 4096) │   (1, 2) │ 1.98728 │   0.993641 │ 0.930621 │ 1.12197 │
│ (4096, 4096) │   (1, 4) │ 3.96338 │   0.990845 │ 0.930621 │ 1.12197 │
│ (4096, 4096) │   (1, 8) │ 7.73237 │   0.966547 │ 0.930621 │ 1.12197 │
│ (4096, 4096) │  (1, 16) │ 14.3881 │   0.899255 │ 0.930336 │ 1.11127 │
│ (4096, 4096) │  (1, 32) │ 25.9754 │   0.811731 │ 0.930336 │ 1.11127 │
│ (4096, 4096) │  (1, 64) │ 50.5666 │   0.790102 │ 0.930421 │ 1.11448 │
│ (4096, 4096) │ (1, 128) │ 94.4984 │   0.738269 │ 0.930421 │ 1.11448 │
└──────────────┴──────────┴─────────┴────────────┴──────────┴─────────┘
```

Slightly differing from the weak scaling results, efficiencies drop below 80% to around 74% at 128 ranks for the strong scaling distributed shallow water model benchmark. This is likely caused by the 128 CPU cores not being sufficiently saturated anymore by the constant `4096^2` grid size thus losing some efficiency overheads.

![ss_shallow_water_efficiency](https://user-images.githubusercontent.com/45054739/129826134-3c526b9f-efd1-436c-9dc1-bde376a035db.png)

### Multi-GPU Shallow Water Model

While still a work in progress, it is possible to use CUDA-aware MPI to run the shallow water model on multiple GPUs. Though efficiencies may not be as high as multi-CPU, the multi-GPU architecture is still worthwhile when keeping in mind the baseline speedups generated by using a single GPU. Note that though it is possible for multiple ranks to share the use of a single GPU, efficiencies would significantly decrease and memory may be insufficient. The results below show up to three ranks each using a separate GPU.

```
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: Linux (powerpc64le-unknown-linux-gnu)
  CPU: unknown
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, pwr9)
Environment:
  JULIA_MPI_PATH = /home/software/spack/openmpi/3.1.4-nhjzelonyovxks5ydtrxehceqxsbf7ik
  JULIA_CUDA_USE_BINARYBUILDER = false
  JULIA_DEPOT_PATH = /nobackup/users/henryguo/projects/henry-test/Oceananigans.jl/benchmark/.julia
  GPU: Tesla V100-SXM2-32GB
```

<html>
<meta charset="UTF-8">
<body>
<table>
  <caption style = "text-align: center;">Shallow water model weak scaling benchmark</caption>
  <tr class = "header headerLastRow">
    <th style = "text-align: right;">size</th>
    <th style = "text-align: right;">ranks</th>
    <th style = "text-align: right;">min</th>
    <th style = "text-align: right;">median</th>
    <th style = "text-align: right;">mean</th>
    <th style = "text-align: right;">max</th>
    <th style = "text-align: right;">memory</th>
    <th style = "text-align: right;">allocs</th>
    <th style = "text-align: right;">samples</th>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 256)</td>
    <td style = "text-align: right;">(1, 1)</td>
    <td style = "text-align: right;">2.702 ms</td>
    <td style = "text-align: right;">2.728 ms</td>
    <td style = "text-align: right;">2.801 ms</td>
    <td style = "text-align: right;">3.446 ms</td>
    <td style = "text-align: right;">2.03 MiB</td>
    <td style = "text-align: right;">5535</td>
    <td style = "text-align: right;">10</td>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 512)</td>
    <td style = "text-align: right;">(1, 2)</td>
    <td style = "text-align: right;">3.510 ms</td>
    <td style = "text-align: right;">3.612 ms</td>
    <td style = "text-align: right;">4.287 ms</td>
    <td style = "text-align: right;">16.546 ms</td>
    <td style = "text-align: right;">2.03 MiB</td>
    <td style = "text-align: right;">5859</td>
    <td style = "text-align: right;">20</td>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 768)</td>
    <td style = "text-align: right;">(1, 3)</td>
    <td style = "text-align: right;">3.553 ms</td>
    <td style = "text-align: right;">3.653 ms</td>
    <td style = "text-align: right;">5.195 ms</td>
    <td style = "text-align: right;">39.152 ms</td>
    <td style = "text-align: right;">2.03 MiB</td>
    <td style = "text-align: right;">5859</td>
    <td style = "text-align: right;">30</td>
  </tr>
</table>
</body>
</html>


<html>
<meta charset="UTF-8">
<body>
<table>
  <caption style = "text-align: center;">Shallow water model weak scaling speedup</caption>
  <tr class = "header headerLastRow">
    <th style = "text-align: right;">size</th>
    <th style = "text-align: right;">ranks</th>
    <th style = "text-align: right;">slowdown</th>
    <th style = "text-align: right;">efficiency</th>
    <th style = "text-align: right;">memory</th>
    <th style = "text-align: right;">allocs</th>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 256)</td>
    <td style = "text-align: right;">(1, 1)</td>
    <td style = "text-align: right;">1.0</td>
    <td style = "text-align: right;">1.0</td>
    <td style = "text-align: right;">1.0</td>
    <td style = "text-align: right;">1.0</td>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 512)</td>
    <td style = "text-align: right;">(1, 2)</td>
    <td style = "text-align: right;">1.32399</td>
    <td style = "text-align: right;">0.755293</td>
    <td style = "text-align: right;">1.00271</td>
    <td style = "text-align: right;">1.05854</td>
  </tr>
  <tr>
    <td style = "text-align: right;">(4096, 768)</td>
    <td style = "text-align: right;">(1, 3)</td>
    <td style = "text-align: right;">1.33901</td>
    <td style = "text-align: right;">0.746818</td>
    <td style = "text-align: right;">1.00271</td>
    <td style = "text-align: right;">1.05854</td>
  </tr>
</table>
</body>
</html>

## Distributed Nonhydrostatic Model

Similar to the distributed shallow water model benchmark results shown above, the distributed nonhydrostatic model was also benchmarked with the strong and weak scaling methods.

### Weak Scaling Nonhydrostatic Model

Weak scaling efficiencies can be improved for the nonhydrostatic model.

```
Oceananigans v0.60.1
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  JULIA_MPI_PATH = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3
  EBVERSIONJULIA = 1.6.1
  JULIA_DEPOT_PATH = :
  JULIA_MPI_BINARY = system
  JULIA_MPI_LIBRARY = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3/lib64/libmpi.so
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1
  JULIA_MPI_ABI = OpenMPI
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1/easybuild/avx2-Core-julia-1.6.1-easybuild-devel
  JULIA_LOAD_PATH = :
  JULIA_MPIEXEC = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3/bin/mpiexec

                                    Nonhydrostatic model weak scaling benchmark
┌──────────────────┬─────────────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│             size │       ranks │        min │     median │       mean │        max │   memory │ allocs │ samples │
├──────────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│   (128, 128, 16) │   (1, 1, 1) │  33.568 ms │  34.087 ms │  34.173 ms │  34.894 ms │ 2.05 MiB │   2762 │      10 │
│   (128, 128, 32) │   (1, 2, 1) │  36.650 ms │  37.161 ms │  37.393 ms │  42.411 ms │ 1.99 MiB │   3096 │      20 │
│   (128, 128, 64) │   (1, 4, 1) │  41.861 ms │  43.440 ms │  46.176 ms │  97.578 ms │ 1.99 MiB │   3136 │      40 │
│  (128, 128, 128) │   (1, 8, 1) │  59.995 ms │  64.110 ms │  68.021 ms │ 138.422 ms │ 1.99 MiB │   3216 │      80 │
│  (128, 128, 256) │  (1, 16, 1) │  62.633 ms │  71.266 ms │  74.775 ms │ 164.206 ms │ 2.01 MiB │   3376 │     160 │
│  (128, 128, 512) │  (1, 32, 1) │ 108.253 ms │ 135.611 ms │ 139.384 ms │ 225.336 ms │ 2.04 MiB │   3722 │     320 │
│ (128, 128, 1024) │  (1, 64, 1) │ 138.504 ms │ 181.043 ms │ 186.386 ms │ 335.170 ms │ 2.12 MiB │   4372 │     640 │
│ (128, 128, 2048) │ (1, 128, 1) │ 218.592 ms │ 285.293 ms │ 290.989 ms │ 434.878 ms │ 2.39 MiB │   5652 │    1280 │
└──────────────────┴─────────────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

                   Nonhydrostatic model weak scaling speedup
┌──────────────────┬─────────────┬──────────┬────────────┬──────────┬─────────┐
│             size │       ranks │  speedup │ efficiency │   memory │  allocs │
├──────────────────┼─────────────┼──────────┼────────────┼──────────┼─────────┤
│   (128, 128, 16) │   (1, 1, 1) │      1.0 │        1.0 │      1.0 │     1.0 │
│   (128, 128, 32) │   (1, 2, 1) │ 0.917292 │   0.917292 │ 0.968543 │ 1.12093 │
│   (128, 128, 64) │   (1, 4, 1) │ 0.784698 │   0.784698 │ 0.969719 │ 1.13541 │
│  (128, 128, 128) │   (1, 8, 1) │ 0.531697 │   0.531697 │ 0.972279 │ 1.16437 │
│  (128, 128, 256) │  (1, 16, 1) │ 0.478315 │   0.478315 │ 0.978143 │  1.2223 │
│  (128, 128, 512) │  (1, 32, 1) │ 0.251361 │   0.251361 │ 0.992878 │ 1.34757 │
│ (128, 128, 1024) │  (1, 64, 1) │ 0.188283 │   0.188283 │  1.03539 │ 1.58291 │
│ (128, 128, 2048) │ (1, 128, 1) │ 0.119482 │   0.119482 │  1.16791 │ 2.04634 │
└──────────────────┴─────────────┴──────────┴────────────┴──────────┴─────────┘
```

![ws_nonhydrostatic_efficiency](https://user-images.githubusercontent.com/45054739/130146112-2dd7e24a-7a79-4000-a899-405362af0f2a.png)


### Strong Scaling Nonhydrostatic Model

Strong scaling efficiencies can also be improved for the nonhydrostatic model.

```
Oceananigans v0.60.1
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  JULIA_MPI_PATH = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3
  EBVERSIONJULIA = 1.6.1
  JULIA_DEPOT_PATH = :
  JULIA_MPI_BINARY = system
  JULIA_MPI_LIBRARY = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3/lib64/libmpi.so
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1
  JULIA_MPI_ABI = OpenMPI
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1/easybuild/avx2-Core-julia-1.6.1-easybuild-devel
  JULIA_LOAD_PATH = :
  JULIA_MPIEXEC = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Compiler/intel2020/openmpi/4.0.3/bin/mpiexec

                                   Nonhydrostatic model strong scaling benchmark
┌─────────────────┬─────────────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│            size │       ranks │        min │     median │       mean │        max │   memory │ allocs │ samples │
├─────────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│ (256, 256, 256) │   (1, 1, 1) │    3.049 s │    3.053 s │    3.053 s │    3.057 s │ 2.05 MiB │   2762 │       2 │
│ (256, 256, 256) │   (1, 2, 1) │    1.609 s │    1.610 s │    1.611 s │    1.620 s │ 1.99 MiB │   3096 │       8 │
│ (256, 256, 256) │   (1, 4, 1) │ 814.290 ms │ 817.305 ms │ 818.685 ms │ 833.792 ms │ 1.99 MiB │   3136 │      28 │
│ (256, 256, 256) │   (1, 8, 1) │ 434.521 ms │ 439.352 ms │ 443.049 ms │ 508.913 ms │ 1.99 MiB │   3216 │      80 │
│ (256, 256, 256) │  (1, 16, 1) │ 251.632 ms │ 272.364 ms │ 277.555 ms │ 370.059 ms │ 2.01 MiB │   3376 │     160 │
│ (256, 256, 256) │  (1, 32, 1) │ 182.380 ms │ 233.322 ms │ 247.325 ms │ 441.971 ms │ 2.04 MiB │   3696 │     320 │
│ (256, 256, 256) │  (1, 64, 1) │ 119.546 ms │ 178.933 ms │ 204.036 ms │ 564.097 ms │ 2.12 MiB │   4346 │     640 │
│ (256, 256, 256) │ (1, 128, 1) │  73.802 ms │ 120.147 ms │ 136.395 ms │ 378.697 ms │ 2.39 MiB │   5626 │    1280 │
└─────────────────┴─────────────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

                 Nonhydrostatic model strong scaling speedup
┌─────────────────┬─────────────┬─────────┬────────────┬──────────┬─────────┐
│            size │       ranks │ speedup │ efficiency │   memory │  allocs │
├─────────────────┼─────────────┼─────────┼────────────┼──────────┼─────────┤
│ (256, 256, 256) │   (1, 1, 1) │     1.0 │        1.0 │      1.0 │     1.0 │
│ (256, 256, 256) │   (1, 2, 1) │ 1.89655 │   0.948276 │ 0.968543 │ 1.12093 │
│ (256, 256, 256) │   (1, 4, 1) │ 3.73522 │   0.933804 │ 0.969719 │ 1.13541 │
│ (256, 256, 256) │   (1, 8, 1) │ 6.94845 │   0.868556 │ 0.972279 │ 1.16437 │
│ (256, 256, 256) │  (1, 16, 1) │ 11.2086 │   0.700536 │ 0.978143 │  1.2223 │
│ (256, 256, 256) │  (1, 32, 1) │ 13.0841 │   0.408879 │ 0.992685 │ 1.33816 │
│ (256, 256, 256) │  (1, 64, 1) │ 17.0612 │   0.266582 │  1.03519 │  1.5735 │
│ (256, 256, 256) │ (1, 128, 1) │  25.409 │   0.198508 │  1.16772 │ 2.03693 │
└─────────────────┴─────────────┴─────────┴────────────┴──────────┴─────────┘
```

![ss_nonhydrostatic_efficiency](https://user-images.githubusercontent.com/45054739/130146219-b354fa25-7d77-4206-8e7e-ec639b2250fa.png)


## Multithreading

Oceananigans can also achieve parallelism via multithreading. Though its efficiencies are less than that of the MPI distributed architectures, its simple setup still makes it a viable option for achieving speedups on simple systems.

### Weak Scaling Multithreaded Shallow Water Model

The initial drop and then rise in efficiencies going from 1 to 2 to 4 threads is likely caused by the 2 threads being automatically allocated onto only one physical CPU core. Though one physical CPU core may contain 2 logical cores each capable of running a separate thread, having 2 threads run on one core will still reduce efficiencies as many resources such as caches and buses must be shared by both threads. Note that there are as many CPU cores allocated as the maximum number of threads.

```
Oceananigans v0.58.9
Julia Version 1.6.0
Commit f9720dc2eb (2021-03-24 12:55 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  EBVERSIONJULIA = 1.6.0
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/easybuild/avx2-Core-julia-1.6.0-easybuild-devel
  JULIA_LOAD_PATH = :

                  Shallow water model weak scaling with multithreading benchmark
┌───────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────────┬─────────┬─────────┐
│          size │ threads │     min │  median │    mean │     max │    memory │  allocs │ samples │
├───────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────────┼─────────┼─────────┤
│   (8192, 512) │       1 │ 1.458 s │ 1.458 s │ 1.458 s │ 1.458 s │  1.37 MiB │    2318 │       4 │
│  (8192, 1024) │       2 │ 2.925 s │ 2.989 s │ 2.989 s │ 3.052 s │ 18.06 MiB │ 1076944 │       2 │
│  (8192, 2048) │       4 │ 2.296 s │ 2.381 s │ 2.397 s │ 2.515 s │ 13.60 MiB │  760190 │       3 │
│  (8192, 4096) │       8 │ 2.347 s │ 2.369 s │ 2.377 s │ 2.415 s │ 16.36 MiB │  891860 │       3 │
│  (8192, 8192) │      16 │ 2.407 s │ 2.548 s │ 2.517 s │ 2.595 s │ 17.44 MiB │  863941 │       3 │
│ (8192, 16384) │      32 │ 3.023 s │ 3.069 s │ 3.069 s │ 3.115 s │ 23.03 MiB │ 1034063 │       2 │
└───────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴───────────┴─────────┴─────────┘

        Shallow water model weak multithreading scaling speedup
┌───────────────┬─────────┬──────────┬────────────┬─────────┬─────────┐
│          size │ threads │ slowdown │ efficiency │  memory │  allocs │
├───────────────┼─────────┼──────────┼────────────┼─────────┼─────────┤
│   (8192, 512) │       1 │      1.0 │        1.0 │     1.0 │     1.0 │
│  (8192, 1024) │       2 │  2.04972 │   0.487872 │ 13.2156 │ 464.601 │
│  (8192, 2048) │       4 │  1.63302 │   0.612363 │ 9.95278 │ 327.951 │
│  (8192, 4096) │       8 │  1.62507 │   0.615359 │ 11.9706 │ 384.754 │
│  (8192, 8192) │      16 │  1.74747 │   0.572257 │  12.755 │  372.71 │
│ (8192, 16384) │      32 │  2.10486 │    0.47509 │  16.846 │ 446.101 │
└───────────────┴─────────┴──────────┴────────────┴─────────┴─────────┘
```

### Strong Scaling Multithreaded Nonhydrostatic Model

The notable and continuous decrease in efficiencies for the strong scaling nonhydrostatic model is likely caused by the `256^3` grid not sufficiently saturating 32 threads running on 32 CPUs. At the time this benchmark was produced, multithreading for both nonhydrostatic and shallow water models is still an active area of improvement. Please use the appropriate scripts found in [`benchmarks`](https://github.com/CliMA/Oceananigans.jl/tree/master/benchmark) to obtain more recent and hopefully ameliorated benchmark results.

```
Oceananigans v0.58.9
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Xeon(R) CPU E5-2683 v4 @ 2.10GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, broadwell)
Environment:
  EBVERSIONJULIA = 1.6.1
  JULIA_DEPOT_PATH = :
  EBROOTJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1
  EBDEVELJULIA = /cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.1/easybuild/avx2-Core-julia-1.6.1-easybuild-devel
  JULIA_LOAD_PATH = :

                                     Multithreading benchmarks
┌──────┬─────────┬────────────┬────────────┬────────────┬────────────┬──────────┬────────┬─────────┐
│ size │ threads │        min │     median │       mean │        max │   memory │ allocs │ samples │
├──────┼─────────┼────────────┼────────────┼────────────┼────────────┼──────────┼────────┼─────────┤
│  256 │       1 │    2.496 s │    2.637 s │    2.637 s │    2.777 s │ 1.70 MiB │   2251 │       2 │
│  256 │       2 │    2.385 s │    2.618 s │    2.618 s │    2.851 s │ 7.03 MiB │ 342397 │       2 │
│  256 │       4 │    1.320 s │    1.321 s │    1.333 s │    1.371 s │ 3.69 MiB │ 113120 │       4 │
│  256 │       8 │ 850.438 ms │ 855.292 ms │ 855.952 ms │ 861.966 ms │ 3.31 MiB │  65709 │       6 │
│  256 │      16 │ 642.225 ms │ 645.458 ms │ 648.685 ms │ 674.259 ms │ 3.60 MiB │  40992 │       8 │
│  256 │      32 │ 680.938 ms │ 694.376 ms │ 701.272 ms │ 746.599 ms │ 4.88 MiB │  36729 │       8 │
└──────┴─────────┴────────────┴────────────┴────────────┴────────────┴──────────┴────────┴─────────┘

     Nonhydrostatic Strong Scaling Multithreading speedup
┌──────┬─────────┬──────────┬────────────┬─────────┬─────────┐
│ size │ threads │ slowdown │ efficiency │  memory │  allocs │
├──────┼─────────┼──────────┼────────────┼─────────┼─────────┤
│  256 │       1 │      1.0 │        1.0 │     1.0 │     1.0 │
│  256 │       2 │ 0.992966 │   0.503542 │ 4.14014 │ 152.109 │
│  256 │       4 │ 0.501089 │   0.498913 │ 2.17724 │ 50.2532 │
│  256 │       8 │ 0.324366 │   0.385367 │ 1.94899 │  29.191 │
│  256 │      16 │ 0.244788 │   0.255323 │ 2.12262 │ 18.2106 │
│  256 │      32 │ 0.263339 │   0.118668 │ 2.87624 │ 16.3167 │
└──────┴─────────┴──────────┴────────────┴─────────┴─────────┘
```

## Tracers

This benchmark tests the performance impacts of running with various amounts of active
and passive tracers and compares the difference in speedup going from CPU to GPU. Number of tracers are listed in the tracers column as (active, passive). 

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

## Older Benchmarks
The following benchmark results are generated from an older version of Oceananigans and with deprecated benchmarking scripts. These legacy benchmark results can still be resonably used as a reference for gauging performance changes across versions of Oceananigans.

### Static ocean

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

### Channel

This benchmark tests the channel model (`topology = (Periodic, Bounded, Bounded)`)
configuration which can be slower due to the use of a more complicated algorithm
(involving 2D cosine transforms) for the pressure solver in the listed version
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
