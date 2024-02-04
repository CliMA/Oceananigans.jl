# Architecture

Passing `CPU()` or `GPU()` to the grid constructor determines whether the grid lives on a CPU or GPU.

Ideally a set up or simulation script does not need to be modified to run on a GPU but still we are smoothing
out rough edges. Generally the CPU wants `Array` objects while the GPU wants `CuArray` objects.

!!! tip "Running on GPUs"
    If you are having issues with running Oceananigans on a GPU, please
    [open an issue](https://github.com/CLiMA/Oceananigans.jl/issues/new) and we'll do our best to help out.
