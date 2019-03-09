# Performance benchmarks

The `benchmarks/benchmarks.jl` script can be run to benchmark Oceananigans.jl on your machine. The script times how long a single time step takes on the CPU and GPU using `Float32` and `Float64` for various model resolutions. It uses TimerOutputs.jl to nicely format the benchmarks. It also prints out CPU->GPU speedups and Float64->Float32 "speedups".

Right now it only benchmarks a simple "static ocean" configuration. The time stepping and Poisson solver still takes the same amount of time whether the ocean is static or active, so it should be indicative of actual performance.

Here is the script's output when run on a single-core of a Intel Xeon E5-2680 v4 @ 2.40 GHz CPU and on an Nvidia Tesla V100 GPU.
```
 ──────────────────────────────────────────────────────────────────────────────────────────────────
             Oceananigans.jl benchmarks                    Time                   Allocations
                                                   ──────────────────────   ───────────────────────
                 Tot / % measured:                       718s / 46.6%           17.2GiB / 0.02%

 Section                                   ncalls     time   %tot     avg     alloc   %tot      avg
 ──────────────────────────────────────────────────────────────────────────────────────────────────
 256x256x256 static ocean (CPU, Float32)       10     168s  50.2%   16.8s   20.3KiB  0.73%  2.03KiB
 256x256x256 static ocean (CPU, Float64)       10     141s  42.3%   14.1s   20.3KiB  0.73%  2.03KiB
 128x128x128 static ocean (CPU, Float32)       10    12.4s  3.72%   1.24s   14.5KiB  0.52%  1.45KiB
 128x128x128 static ocean (CPU, Float64)       10    9.00s  2.69%   900ms   14.8KiB  0.54%  1.48KiB
  64x 64x 64 static ocean (CPU, Float32)       10    1.03s  0.31%   103ms   14.2KiB  0.51%  1.42KiB
 256x256x256 static ocean (GPU, Float64)       10    891ms  0.27%  89.1ms    333KiB  12.0%  33.3KiB
 256x256x256 static ocean (GPU, Float32)       10    859ms  0.26%  85.9ms    329KiB  11.9%  32.9KiB
  64x 64x 64 static ocean (CPU, Float64)       10    635ms  0.19%  63.5ms   13.5KiB  0.49%  1.35KiB
 128x128x128 static ocean (GPU, Float64)       10   80.2ms  0.02%  8.02ms    332KiB  12.0%  33.2KiB
 128x128x128 static ocean (GPU, Float32)       10   77.0ms  0.02%  7.70ms    329KiB  11.9%  32.9KiB
  32x 32x 32 static ocean (CPU, Float32)       10   72.3ms  0.02%  7.23ms   13.1KiB  0.47%  1.31KiB
  32x 32x 32 static ocean (CPU, Float64)       10   45.1ms  0.01%  4.51ms   13.5KiB  0.49%  1.35KiB
  64x 64x 64 static ocean (GPU, Float64)       10   8.30ms  0.00%   830μs    332KiB  12.0%  33.2KiB
  64x 64x 64 static ocean (GPU, Float32)       10   8.05ms  0.00%   805μs    329KiB  11.9%  32.9KiB
  32x 32x 32 static ocean (GPU, Float64)       10   3.63ms  0.00%   363μs    332KiB  12.0%  33.2KiB
  32x 32x 32 static ocean (GPU, Float32)       10   3.45ms  0.00%   345μs    329KiB  11.9%  32.9KiB
 ──────────────────────────────────────────────────────────────────────────────────────────────────

CPU Float64 -> Float32 speedups:
 32x 32x 32 static ocean: 0.623
 64x 64x 64 static ocean: 0.614
128x128x128 static ocean: 0.723
256x256x256 static ocean: 0.841

GPU Float64 -> Float32 speedups:
 32x 32x 32 static ocean: 1.052
 64x 64x 64 static ocean: 1.031
128x128x128 static ocean: 1.042
256x256x256 static ocean: 1.038

CPU -> GPU speedsup:
 32x 32x 32 static ocean (Float32): 20.923
 32x 32x 32 static ocean (Float64): 12.402
 64x 64x 64 static ocean (Float32): 128.536
 64x 64x 64 static ocean (Float64): 76.582
128x128x128 static ocean (Float32): 161.689
128x128x128 static ocean (Float64): 112.144
256x256x256 static ocean (Float32): 195.877
256x256x256 static ocean (Float64): 158.772
```
