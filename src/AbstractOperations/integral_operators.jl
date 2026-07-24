using Oceananigans.Fields: Field

#####
##### Unicode integral operators
#####

function integral_field(field, dims; cumulative=false, kwargs...)
    integral = cumulative ? CumulativeIntegral(field; dims, kwargs...) :
                            Integral(field; dims, kwargs...)

    return Field(integral)
end

"""
    ∫dx(field; cumulative=false, reverse=false, condition=nothing, mask=0)

Return a `Field` holding ``∫ f \\, dx``, the integral of `field` over ``x``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫dx(field)`.

With `cumulative=true` the running integral ``∫^x f \\, dx'`` computed by
[`CumulativeIntegral`](@ref) is returned instead, and `reverse=true` accumulates it westward
from the eastern edge of the domain.

See also [`∫dy`](@ref), [`∫dz`](@ref), [`∫∫dydz`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.
"""
∫dx(field::AbstractField; kwargs...) = integral_field(field, 1; kwargs...)

"""
    ∫dy(field; cumulative=false, reverse=false, condition=nothing, mask=0)

Return a `Field` holding ``∫ f \\, dy``, the integral of `field` over ``y``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫dy(field)`.

With `cumulative=true` the running integral ``∫^y f \\, dy'`` computed by
[`CumulativeIntegral`](@ref) is returned instead, and `reverse=true` accumulates it southward
from the northern edge of the domain.

See also [`∫dx`](@ref), [`∫dz`](@ref), [`∫∫dxdz`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.
"""
∫dy(field::AbstractField; kwargs...) = integral_field(field, 2; kwargs...)

"""
    ∫dz(field; cumulative=false, reverse=false, condition=nothing, mask=0)

Return a `Field` holding ``∫ f \\, dz``, the integral of `field` over ``z``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫dz(field)`.

With `cumulative=true` the running integral ``∫^z f \\, dz'`` computed by
[`CumulativeIntegral`](@ref) is returned instead, and `reverse=true` accumulates it downward
from the top of the domain.

See also [`∫dx`](@ref), [`∫dy`](@ref), [`∫∫dxdy`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.

Example
=======

Integrate ``f(z) = z`` over ``z ∈ [0, 1]``, both in full and cumulatively.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=8, z=(0, 1), topology=(Flat, Flat, Bounded));

julia> c = CenterField(grid);

julia> set!(c, z -> z);

julia> ∫dz(c)[1, 1, 1]
0.5

julia> ∫dz(c, cumulative=true)
1×1×9 Field{Center, Center, Face} on RectilinearGrid on CPU
├── data: OffsetArrays.OffsetArray{Float64, 3, Array{Float64, 3}}, size: (1, 1, 9)
├── grid: 1×1×8 RectilinearGrid{Float64, Flat, Flat, Bounded} on CPU with 0×0×3 halo
├── operand: CumulativeIntegral of BinaryOperation at (Center, Center, Center) over dims 3
├── status: time=0.0
└── data: 1×1×15 OffsetArray(::Array{Float64, 3}, 1:1, 1:1, -2:12) with eltype Float64 with indices 1:1×1:1×-2:12
    └── max=0.5, min=0.0, mean=0.177083
```
"""
∫dz(field::AbstractField; kwargs...) = integral_field(field, 3; kwargs...)

"""
    ∫∫dxdy(field; condition=nothing, mask=0)

Return a `Field` holding ``∬ f \\, dx \\, dy``, the integral of `field` over ``x`` and ``y``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫∫dxdy(field)`.

See also [`∫∫dxdz`](@ref), [`∫∫dydz`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.
"""
∫∫dxdy(field::AbstractField; kwargs...) = integral_field(field, (1, 2); kwargs...)

"""
    ∫∫dxdz(field; condition=nothing, mask=0)

Return a `Field` holding ``∬ f \\, dx \\, dz``, the integral of `field` over ``x`` and ``z``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫∫dxdz(field)`.

See also [`∫∫dxdy`](@ref), [`∫∫dydz`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.
"""
∫∫dxdz(field::AbstractField; kwargs...) = integral_field(field, (1, 3); kwargs...)

"""
    ∫∫dydz(field; condition=nothing, mask=0)

Return a `Field` holding ``∬ f \\, dy \\, dz``, the integral of `field` over ``y`` and ``z``.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫∫dydz(field)`.

See also [`∫∫dxdy`](@ref), [`∫∫dxdz`](@ref), and [`∫∫∫dxdydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.
"""
∫∫dydz(field::AbstractField; kwargs...) = integral_field(field, (2, 3); kwargs...)

"""
    ∫∫∫dxdydz(field; condition=nothing, mask=0)

Return a `Field` holding ``∭ f \\, dx \\, dy \\, dz``, the integral of `field` over the
three-dimensional domain. [`∫dV`](@ref) is an alias for `∫∫∫dxdydz`.

The returned `Field` is computed on construction and, unlike the `Reduction` returned by
[`Integral`](@ref), may be used within other abstract operations, as in `field / ∫dV(field)`.

See also [`∫∫dxdy`](@ref), [`∫∫dxdz`](@ref), and [`∫∫dydz`](@ref);
[`ConditionalOperation`](@ref Oceananigans.AbstractOperations.ConditionalOperation)
documents the `condition` and `mask` keyword arguments.

Example
=======

Compute the integral of ``f(x, y, z) = x y z`` over the domain
``(x, y, z) ∈ [0, 1] × [0, 1] × [0, 1]``. The analytical answer
is ``∭ x y z \\, dx \\, dy \\, dz = 1/8``.

```jldoctest
julia> using Oceananigans

julia> grid = RectilinearGrid(size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1));

julia> f = CenterField(grid);

julia> set!(f, (x, y, z) -> x * y * z);

julia> ∫dV(f)[1, 1, 1]
0.125
```
"""
∫∫∫dxdydz(field::AbstractField; kwargs...) = integral_field(field, (1, 2, 3); kwargs...)

"""
    ∫dV(field; condition=nothing, mask=0)

Alias for [`∫∫∫dxdydz`](@ref): return a `Field` holding ``∭ f \\, dx \\, dy \\, dz``, the
integral of `field` over the three-dimensional domain.
"""
const ∫dV = ∫∫∫dxdydz
