using Adapt

using Oceananigans.Fields: interpolate
using Oceananigans.Grids: ξnode, ηnode, znode
using Oceananigans.Utils: Utils

"""
    struct InterpolatingKernelFunction

Callable kernel function underlying an [`InterpolatedOperation`](@ref). Interpolates `operand`
to the coordinates specified by `x`, `y`, `z`: a `Number` interpolates to that (native grid)
coordinate, while `nothing` keeps `operand`'s own node in that direction.
"""
struct InterpolatingKernelFunction{O, L, X, Y, Z}
    operand :: O
    loc :: L        # operand's (ℓx, ℓy, ℓz) location instances
    x :: X          # a coordinate to interpolate to, or `nothing` to keep the operand's node
    y :: Y
    z :: Z
end

@inline target_or_node(::Nothing, node) = node   # unspecified direction: evaluate at operand's node
@inline target_or_node(target, node) = target    # specified direction: interpolate to `target`

@inline function (kf::InterpolatingKernelFunction)(i, j, k, grid)
    ℓx, ℓy, ℓz = kf.loc
    x = target_or_node(kf.x, ξnode(i, j, k, grid, ℓx, ℓy, ℓz))
    y = target_or_node(kf.y, ηnode(i, j, k, grid, ℓx, ℓy, ℓz))
    z = target_or_node(kf.z, znode(i, j, k, grid, ℓx, ℓy, ℓz))
    return interpolate((x, y, z), kf.operand, kf.loc, grid)
end

Adapt.adapt_structure(to, kf::InterpolatingKernelFunction) =
    InterpolatingKernelFunction(adapt(to, kf.operand), kf.loc, kf.x, kf.y, kf.z)

function Utils.prettysummary(kf::InterpolatingKernelFunction)
    targets = String[]
    isnothing(kf.x) || push!(targets, "x=$(kf.x)")
    isnothing(kf.y) || push!(targets, "y=$(kf.y)")
    isnothing(kf.z) || push!(targets, "z=$(kf.z)")
    return string("InterpolatingKernelFunction(", join(targets, ", "), ")")
end

const InterpolatedOperation{LX, LY, LZ} =
    KernelFunctionOperation{LX, LY, LZ, <:Any, <:Any, <:InterpolatingKernelFunction} where {LX, LY, LZ}

"""
$(TYPEDSIGNATURES)

Return a `KernelFunctionOperation` that interpolates `operand` (a `Field` or `AbstractOperation`)
to fixed coordinates in the directions selected by the keyword arguments `x`, `y`, `z`.

Each specified direction is interpolated to the given coordinate and *reduced* (its location
becomes `Nothing`); unspecified directions keep `operand`'s location and are evaluated at
`operand`'s own nodes. For example `InterpolatedOperation(c; z=100)` is a two-dimensional
operation at `(Center, Center, Nothing)`, whereas `InterpolatedOperation(c; x=1, y=2)` is a
vertical column at `(Nothing, Nothing, Center)`.

Coordinates are the grid's native coordinates: `x, y` are `longitude, latitude` on curvilinear
grids (`LatitudeLongitudeGrid`, `OrthogonalSphericalShellGrid`) and Cartesian `x, y` on
`RectilinearGrid`; `z` is the vertical coordinate on every grid.

Example
=======

```jldoctest itp_op
using Oceananigans
using Oceananigans.AbstractOperations: InterpolatedOperation

grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
c = CenterField(grid)

c_at_z = InterpolatedOperation(c; z=-0.5)

# output
KernelFunctionOperation at (Center, Center, ⋅)
├── grid: 4×4×4 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×3 halo
├── kernel_function: InterpolatingKernelFunction(z=-0.5)
└── arguments: ()
```
"""
function InterpolatedOperation(operand::AbstractField; x=nothing, y=nothing, z=nothing)
    (isnothing(x) & isnothing(y) & isnothing(z)) &&
        throw(ArgumentError("InterpolatedOperation requires at least one of x, y, z"))

    validate_operand(operand)

    grid = operand.grid
    FT = eltype(grid)
    LX, LY, LZ = location(operand)

    X = isnothing(x) ? nothing : convert(FT, x)
    Y = isnothing(y) ? nothing : convert(FT, y)
    Z = isnothing(z) ? nothing : convert(FT, z)

    kernel_function = InterpolatingKernelFunction(operand, (LX(), LY(), LZ()), X, Y, Z)

    ℓx = isnothing(x) ? LX : Nothing
    ℓy = isnothing(y) ? LY : Nothing
    ℓz = isnothing(z) ? LZ : Nothing

    return KernelFunctionOperation{ℓx, ℓy, ℓz}(kernel_function, grid)
end
