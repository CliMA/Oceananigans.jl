struct KernelFunctionOperation{LX, LY, LZ, P, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
    kernel_function :: K
    computed_dependencies :: D
    parameters :: P
    grid :: G

    function KernelFunctionOperation{LX, LY, LZ}(kernel_function::K, computed_dependencies::D,
                                                 parameters::P, grid::G) where {LX, LY, LZ, K, G, D, P}
        T = eltype(grid)
        return new{LX, LY, LZ, P, G, T, K, D}(kernel_function, computed_dependencies, parameters, grid)
    end

end


"""
    KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid;
                                        computed_dependencies=(), parameters=nothing)

Constructs a `KernelFunctionOperation` at location `(LX, LY, LZ)` on `grid` an with
an optional iterable of `computed_dependencies` and arbitrary `parameters`.

With `isnothing(parameters)` (the default), `kernel_function` is called with

```julia
kernel_function(i, j, k, grid, computed_dependencies...)
```

Otherwise `kernel_function` is called with

```julia
kernel_function(i, j, k, grid, computed_dependencies..., parameters)
```

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
using Oceananigans.Operators: ζ₃ᶠᶠᵃ # called with signature ζ₃ᶠᶠᵃ(i, j, k, grid, u, v)

grid = model.grid
u, v, w = model.velocities

ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᵃ, grid, computed_dependencies=(u, v))
```
"""
function KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid;
                                            computed_dependencies = (),
                                            parameters = nothing) where {LX, LY, LZ}

    return KernelFunctionOperation{LX, LY, LZ}(kernel_function, computed_dependencies, parameters, grid)
end

@inline Base.getindex(κ::KernelFunctionOperation, i, j, k) = κ.kernel_function(i, j, k, κ.grid, κ.computed_dependencies..., κ.parameters)
@inline Base.getindex(κ::KernelFunctionOperation{LX, LY, LZ, <:Nothing}, i, j, k) where {LX, LY, LZ} = κ.kernel_function(i, j, k, κ.grid, κ.computed_dependencies...)

# Compute dependencies
compute_at!(κ::KernelFunctionOperation, time) = Tuple(compute_at!(d, time) for d in κ.computed_dependencies)

"Adapt `KernelFunctionOperation` to work on the GPU via CUDAnative and CUDAdrv."
Adapt.adapt_structure(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(Adapt.adapt(to, κ.kernel_function),
                                        Adapt.adapt(to, κ.computed_dependencies),
                                        Adapt.adapt(to, κ.parameters),
                                        Adapt.adapt(to, κ.grid))

