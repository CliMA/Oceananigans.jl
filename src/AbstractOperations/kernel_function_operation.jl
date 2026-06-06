using Oceananigans.Utils: Utils, shortsummary, construct_regionally, prettysummary

"""
    KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, arguments...)

Construct a `KernelFunctionOperation` at location `(LX, LY, LZ)` on `grid` with `arguments`.

`kernel_function` is called with

```julia
kernel_function(i, j, k, grid, arguments...)
```

If the location contains `Nothing`, `kernel_function` may also omit the indices of the
`Nothing` dimensions: for example, at `(Center, Center, Nothing)` it may be called with

```julia
kernel_function(i, j, grid, arguments...)
```

Note that `compute!(kfo::KernelFunctionOperation)` calls `compute!` on all `kfo.arguments`.

Examples
========

Construct a `KernelFunctionOperation` that returns random numbers:

```jldoctest kfo
using Oceananigans

grid = RectilinearGrid(size=(1, 8, 8), extent=(1, 1, 1));

random_kernel_function(i, j, k, grid) = rand();
kernel_op = KernelFunctionOperation{Center, Center, Center}(random_kernel_function, grid)

# output
KernelFunctionOperation at (Center, Center, Center)
├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×3×3 halo
├── kernel_function: random_kernel_function (generic function with 1 method)
└── arguments: ()
```

Construct a `KernelFunctionOperation` using the vertical vorticity operator used internally
to compute vertical vorticity on all grids:

```jldoctest kfo
using Oceananigans.Operators: ζ₃ᶠᶠᶜ # called with signature ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)

model = HydrostaticFreeSurfaceModel(grid)
u, v, w = model.velocities
ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid, u, v)

# output
KernelFunctionOperation at (Face, Face, Center)
├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×3×3 halo
├── kernel_function: ζ₃ᶠᶠᶜ (generic function with 1 method)
└── arguments: ("Field", "Field")
```

Construct a `KernelFunctionOperation` at a reduced location using a kernel function
that omits the index of the `Nothing` dimension:

```jldoctest kfo
surface_kernel_function(i, j, grid) = i + j
surface_op = KernelFunctionOperation{Center, Center, Nothing}(surface_kernel_function, grid)

# output
KernelFunctionOperation at (Center, Center, ⋅)
├── grid: 1×8×8 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×3×3 halo
├── kernel_function: surface_kernel_function (generic function with 1 method)
└── arguments: ()
```
"""
struct KernelFunctionOperation{LX, LY, LZ, G, T, K, D} <: AbstractOperation{LX, LY, LZ, G, T}
    kernel_function :: K
    grid :: G
    arguments :: D

    function KernelFunctionOperation{LX, LY, LZ}(kernel_function::K, grid::G, arguments::D,
                                                 ::Type{T}=eltype(grid)) where {LX, LY, LZ, G, T, K, D<:Tuple}
        return new{LX, LY, LZ, G, T, K, D}(kernel_function, grid, arguments)
    end

end

# Convenience outer constructor: splat arguments into a tuple.
# T defaults to eltype(grid) via the inner constructor.
function KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, arguments...) where {LX, LY, LZ}
    kernel_function = possibly_reduced_kernel_function(kernel_function, (LX, LY, LZ), arguments)
    return KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, tuple(arguments...))
end

"""
    ReducedKernelFunction{D, F}

Wrap a kernel function defined at a reduced location, forwarding only the indices of the non-`Nothing` dimensions `D`, 
so that `kernel_function(i, j, k, grid, args...)` calls, e.g. for `D = (1, 2)`, `kernel_function(i, j, grid, args...)`.
"""
struct ReducedKernelFunction{Dims, F}
    kernel_function :: F
    ReducedKernelFunction{Dims}(kernel_function::F) where {Dims, F} = new{Dims, F}(kernel_function)
end

@inline (rkf::ReducedKernelFunction{Dims})(i, j, k, grid, arguments...) where Dims = rkf.kernel_function(map(d -> (i, j, k)[d], Dims)..., grid, arguments...)

function possibly_reduced_kernel_function(kernel_function, location, arguments)
    kept_dimensions = Tuple(d for d in 1:3 if location[d] !== Nothing)
    if length(kept_dimensions) == 3
        return kernel_function
    end
    reduced_parameter_count = length(kept_dimensions) + 1 + length(arguments)
    has_reduced_method = any(m -> !m.isva && m.nargs - 1 == reduced_parameter_count, methods(kernel_function))
    return has_reduced_method ? ReducedKernelFunction{kept_dimensions}(kernel_function) : kernel_function
end

Adapt.adapt_structure(to, rkf::ReducedKernelFunction{Dims}) where Dims = ReducedKernelFunction{Dims}(Adapt.adapt(to, rkf.kernel_function))

Utils.prettysummary(rkf::ReducedKernelFunction) = prettysummary(rkf.kernel_function)

@inline Base.getindex(κ::KernelFunctionOperation, i, j, k) = κ.kernel_function(i, j, k, κ.grid, κ.arguments...)
indices(κ::KernelFunctionOperation) = construct_regionally(intersect_indices, location(κ), κ.arguments...)
compute_at!(κ::KernelFunctionOperation, time) = Tuple(compute_at!(d, time) for d in κ.arguments)

"Adapt `KernelFunctionOperation` to work on the GPU via KernelAbstractions."
Adapt.adapt_structure(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(Adapt.adapt(to, κ.kernel_function),
                                        Adapt.adapt(to, κ.grid),
                                        Tuple(Adapt.adapt(to, a) for a in κ.arguments),
                                        eltype(κ))

Architectures.on_architecture(to, κ::KernelFunctionOperation{LX, LY, LZ}) where {LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ}(on_architecture(to, κ.kernel_function),
                                        on_architecture(to, κ.grid),
                                        Tuple(on_architecture(to, a) for a in κ.arguments),
                                        eltype(κ))

Base.show(io::IO, kfo::KernelFunctionOperation) =
    print(io,
      summary(kfo), '\n',
      "├── grid: ", summary(kfo.grid), '\n',
      "├── kernel_function: ", prettysummary(kfo.kernel_function), '\n',
      "└── arguments: ", if isempty(kfo.arguments)
                             "()"
                         else
                             # Tuple(string(prettysummary(a)) for a in kfo.arguments[1:end-1])...,
                             # prettysummary(kfo.arguments[end])
                             Tuple(shortsummary(a) for a in kfo.arguments[1:end-1])...,
                             shortsummary(kfo.arguments[end])
                         end
    )
