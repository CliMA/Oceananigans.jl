# Performance benchmarks

The `benchmarks/benchmarks.jl` script can be run to benchmark Oceananigans.jl on your machine. The script times how long a single time step takes on the CPU and GPU using `Float32` and `Float64` for various model resolutions. It uses TimerOutputs.jl to nicely format the benchmarks. It also prints out CPU->GPU speedups and Float64->Float32 "speedups".

Right now it only benchmarks a simple "static ocean" configuration. The time stepping and Poisson solver still takes the same amount of time whether the ocean is static or active, so it should be indicative of actual performance.

Here is the script's output when run on a single-core of a Intel Xeon E5-2680 v4 @ 2.40 GHz CPU and on an Nvidia Tesla V100 GPU.
```
──────────────────────────────────────────────────────────────────────────────────────────────────
            Oceananigans.jl benchmarks                    Time                   Allocations      
                                                  ──────────────────────   ───────────────────────
                Tot / % measured:                       227s / 45.6%           18.7GiB / 0.06%    

Section                                   ncalls     time   %tot     avg     alloc   %tot      avg
──────────────────────────────────────────────────────────────────────────────────────────────────
256x256x256 static ocean (CPU, Float32)       10    54.4s  52.5%   5.44s   60.0KiB  0.48%  6.00KiB
256x256x256 static ocean (CPU, Float64)       10    36.9s  35.6%   3.69s   77.8KiB  0.62%  7.78KiB
128x128x128 static ocean (CPU, Float32)       10    6.38s  6.16%   638ms   60.0KiB  0.48%  6.00KiB
128x128x128 static ocean (CPU, Float64)       10    4.04s  3.90%   404ms   77.8KiB  0.62%  7.78KiB
 64x 64x 64 static ocean (CPU, Float32)       10    748ms  0.72%  74.8ms   60.0KiB  0.48%  6.00KiB
 64x 64x 64 static ocean (CPU, Float64)       10    412ms  0.40%  41.2ms   77.8KiB  0.62%  7.78KiB
256x256x256 static ocean (GPU, Float64)       10    284ms  0.27%  28.4ms   1.59MiB  12.9%   163KiB
256x256x256 static ocean (GPU, Float32)       10    243ms  0.23%  24.3ms   1.35MiB  11.0%   139KiB
 32x 32x 32 static ocean (CPU, Float32)       10   80.3ms  0.08%  8.03ms   60.0KiB  0.48%  6.00KiB
 32x 32x 32 static ocean (CPU, Float64)       10   45.2ms  0.04%  4.52ms   77.8KiB  0.62%  7.78KiB
128x128x128 static ocean (GPU, Float64)       10   35.9ms  0.03%  3.59ms   1.59MiB  12.9%   163KiB
128x128x128 static ocean (GPU, Float32)       10   32.3ms  0.03%  3.23ms   1.35MiB  11.0%   139KiB
 64x 64x 64 static ocean (GPU, Float64)       10   6.54ms  0.01%   654μs   1.59MiB  12.9%   163KiB
 64x 64x 64 static ocean (GPU, Float32)       10   6.14ms  0.01%   614μs   1.35MiB  11.0%   139KiB
 32x 32x 32 static ocean (GPU, Float64)       10   5.77ms  0.01%   577μs   1.59MiB  12.9%   163KiB
 32x 32x 32 static ocean (GPU, Float32)       10   5.68ms  0.01%   568μs   1.35MiB  11.0%   139KiB
──────────────────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedup:
32x 32x 32 static ocean: 0.562
64x 64x 64 static ocean: 0.551
128x128x128 static ocean: 0.633
256x256x256 static ocean: 0.677

GPU Float64 -> Float32 speedup:
32x 32x 32 static ocean: 1.015
64x 64x 64 static ocean: 1.066
128x128x128 static ocean: 1.114
256x256x256 static ocean: 1.167

CPU -> GPU speedup:
32x 32x 32 static ocean (Float32): 14.138
32x 32x 32 static ocean (Float64): 7.829
64x 64x 64 static ocean (Float32): 121.806
64x 64x 64 static ocean (Float64): 62.924
128x128x128 static ocean (Float32): 197.906
128x128x128 static ocean (Float64): 112.417
256x256x256 static ocean (Float32): 223.748
256x256x256 static ocean (Float64): 129.923
```
