struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
    kernel_function :: K
    arguments :: D
    grid :: G

    function KernelFunctionOperation{LX, LY, LZ}(kernel_function::K,
                                                 arguments::D,
                                                 grid::G) where {LX, LY, LZ, K, G, D, P}
        T = eltype(grid)
        return new{LX, LY, LZ, P, G, T, K, D}(kernel_function, arguments, grid)
    end

end


"""
    KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid; arguments=())

Construct a `KernelFunctionOperation` at location `(LX, LY, LZ)` on `grid` with
`arguments`.

`kernel_function` is called with

```julia
kernel_function(i, j, k, grid, arguments...)
```

Note that `compute!(kfo::KernelFunctionOperation)` calls `compute!` on
all `kfo.arguments`.

Examples
========

Construct a kernel function operation that returns random numbers:

```julia
random_kernel_function(i, j, k, grid) = rand() # use CUDA.rand on the GPU

kernel_op = KernelFunctionOperation{Center, Center, Center}(random_kernel_function, grid)
```

Construct a kernel function operation using the vertical vorticity operator
valid on curvilinear and cubed sphere grids:

```julia
using Oceananigans.Operators: ζ₃ᶠᶠᶜ # called with signature ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

grid = model.grid
u, v, w = model.velocities

ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, arguments=(u, v))
```
"""
KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid; arguments::Tuple = ()) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(kernel_function, arguments, grid)

indices(κ::KernelFunctionOperation) = interpolate_indices(κ.arguments...; loc_operation = location(κ))

@inline Base.getindex(κ::KernelFunctionOperation, i, j, k) = κ.kernel_function(i, j, k, κ.grid, κ.arguments...)

# Compute dependencies
compute_at!(κ::KernelFunctionOperation, time) = Tuple(compute_at!(d, time) for d in κ.arguments)

"Adapt `KernelFunctionOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(Adapt.adapt(to, κ.kernel_function),
                                        Adapt.adapt(to, κ.arguments),
                                        Adapt.adapt(to, κ.grid))
