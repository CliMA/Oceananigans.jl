# Architecture

Passing `CPU()` or `GPU()` to the grid constructor determines whether the grid lives on a CPU or GPU.

Ideally a set up or simulation script does not need to be modified to run on a GPU but still we are smoothing
out rough edges. Generally the CPU wants `Array` objects while the GPU wants `CuArray` objects.

!!! tip "Running on GPUs"
    The section on [simulation tips](@ref simulation_tips) includes information that can come handy
    when running on GPUs.
    
    We would very welcome any suggestions you may have to improve the API and make transitions
    from CPU to GPU even smoother. Please 
    [open an issue](https://github.com/CLiMA/Oceananigans.jl/issues/new) with any such suggestions.
