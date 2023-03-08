struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
    kernel_function :: K
    arguments :: D
    grid :: G

    @doc"""
        KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, arguments...)
    
    Construct a `KernelFunctionOperation` at location `(LX, LY, LZ)` on `grid` with `arguments`.
    
    `kernel_function` is called with
    
    ```julia
    kernel_function(i, j, k, grid, arguments...)
    ```
    
    Note that `compute!(kfo::KernelFunctionOperation)` calls `compute!` on all `kfo.arguments`.
    
    Examples
    ========
    
    Construct a KernelFunctionOperation that returns random numbers:
    
    ```julia
    random_kernel_function(i, j, k, grid) = rand() # use CUDA.rand on the GPU
    
    kernel_op = KernelFunctionOperation{Center, Center, Center}(random_kernel_function, grid)
    ```
    
    Construct a KernelFunctionOperation using the vertical vorticity operator
    used internally to compute vertical vorticity on all grids:
    
    ```julia
    using Oceananigans.Operators: ζ₃ᶠᶠᶜ # called with signature ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
    
    grid = model.grid
    u, v, w = model.velocities
    
    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)
    ```
    """
    function KernelFunctionOperation{LX, LY, LZ}(kernel_function::K,
                                                 grid::G, 
                                                 arguments::D...) where {LX, LY, LZ, K, G, D}
        T = eltype(grid)
        return new{LX, LY, LZ, G, T, K, D}(kernel_function, arguments, grid)
    end

end

@inline Base.getindex(κ::KernelFunctionOperation, i, j, k) = κ.kernel_function(i, j, k, κ.grid, κ.arguments...)
indices(κ::KernelFunctionOperation) = interpolate_indices(κ.arguments...; loc_operation = location(κ))
compute_at!(κ::KernelFunctionOperation, time) = Tuple(compute_at!(d, time) for d in κ.arguments)

"Adapt `KernelFunctionOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(Adapt.adapt(to, κ.kernel_function),
                                        Adapt.adapt(to, κ.arguments),
                                        Adapt.adapt(to, κ.grid))

