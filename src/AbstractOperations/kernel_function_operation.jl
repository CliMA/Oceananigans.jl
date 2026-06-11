using Oceananigans.Utils: shortsummary, construct_regionally, prettysummary

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

The full three-index call is always preferred when it is applicable, so a function that
already accepts `(i, j, k, grid, arguments...)` keeps that behavior at every location; the
reduced call is used only when the full one is not applicable.

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
    return KernelFunctionOperation{LX, LY, LZ}(kernel_function, grid, tuple(arguments...))
end

# `getindex` calls the kernel function with the full `(i, j, k, grid, args...)` signature
# whenever that call is applicable. At a reduced location it otherwise drops the indices of
# the `Nothing` dimensions, calling e.g. `kernel_function(i, j, grid, args...)`
@inline function Base.getindex(κ::KernelFunctionOperation{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    if applicable(κ.kernel_function, i, j, k, κ.grid, κ.arguments...)
        return κ.kernel_function(i, j, k, κ.grid, κ.arguments...)
    else
        reduced_indices = (kept_index(LX, i)..., kept_index(LY, j)..., kept_index(LZ, k)...)
        return κ.kernel_function(reduced_indices..., κ.grid, κ.arguments...)
    end
end

@inline kept_index(::Type{Nothing}, index) = ()
@inline kept_index(::Type, index) = (index,)

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
