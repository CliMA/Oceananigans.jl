import Adapt
using AMDGPU: ROCArray

"""
    struct BoundaryCondition{C<:AbstractBoundaryConditionClassification, T}

Container for boundary conditions.
"""
struct BoundaryCondition{C<:AbstractBoundaryConditionClassification, T}
    classification :: C
    condition :: T
end

"""
    BoundaryCondition(Classification::DataType, condition)

Construct a boundary condition of type `BC` with a number or array as a `condition`.

Boundary condition types include `Periodic`, `Flux`, `Value`, `Gradient`, and `Open`.
"""
BoundaryCondition(Classification::DataType, condition) = BoundaryCondition(Classification(), condition)

"""
    BoundaryCondition(Classification::DataType, condition::Function;
                      parameters = nothing,
                      discrete_form = false,
                      field_dependencies=())

Construct a boundary condition of type `Classification` with a function boundary `condition`.

By default, the function boudnary `condition` is assumed to have the 'continuous form'
`condition(ξ, η, t)`, where `t` is time and `ξ` and `η` vary along the boundary.
In particular:

- On `x`-boundaries, `condition(y, z, t)`.
- On `y`-boundaries, `condition(x, z, t)`.
- On `z`-boundaries, `condition(x, y, t)`.

If `parameters` is not `nothing`, then function boundary conditions have the form
`func(ξ, η, t, parameters)`, where `ξ` and `η` are spatial coordinates varying along
the boundary as explained above.

If `discrete_form = true`, the function `condition` is assumed to have the "discrete form",
```
condition(i, j, grid, clock, model_fields)
```
where `i`, and `j` are indices that vary along the boundary. If `discrete_form = true` and
`parameters` is not `nothing`, the function `condition` is called with
```
condition(i, j, grid, clock, model_fields, parameters)
```
"""
function BoundaryCondition(Classification::DataType, condition::Function;
                           parameters = nothing,
                           discrete_form = false,
                           field_dependencies=())

    if discrete_form
        field_dependencies != () && error("Cannot set `field_dependencies` when `discrete_form=true`!")
        condition = DiscreteBoundaryFunction(condition, parameters)
    else
        # The `ContinuousBoundaryFunction` is "regularized" for field location in the FieldBoundaryConditions constructor.
        condition = ContinuousBoundaryFunction(condition, parameters, field_dependencies)
    end

    return BoundaryCondition{Classification, typeof(condition)}(Classification(), condition)
end

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BoundaryCondition{C, A}) where {C<:AbstractBoundaryConditionClassification, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

#####
##### Some abbreviations to make life easier.
#####

# These type aliases make dispatching on BCs easier (not exported).
const BC   = BoundaryCondition
const FBC  = BoundaryCondition{<:Flux}
const PBC  = BoundaryCondition{<:Periodic}
const OBC  = BoundaryCondition{<:Open}
const VBC  = BoundaryCondition{<:Value}
const GBC  = BoundaryCondition{<:Gradient}
const ZFBC = BoundaryCondition{Flux, Nothing} # "zero" flux
const CBC  = BoundaryCondition{<:Communication}

# More readable BC constructors for the public API.
    PeriodicBoundaryCondition()  = BoundaryCondition(Periodic,      nothing)
      NoFluxBoundaryCondition()  = BoundaryCondition(Flux,          nothing)
ImpenetrableBoundaryCondition()  = BoundaryCondition(Open,          nothing)
CommunicationBoundaryCondition() = BoundaryCondition(Communication, nothing)

     FluxBoundaryCondition(val; kwargs...)     = BoundaryCondition(Flux, val; kwargs...)
    ValueBoundaryCondition(val; kwargs...)     = BoundaryCondition(Value, val; kwargs...)
 GradientBoundaryCondition(val; kwargs...)     = BoundaryCondition(Gradient, val; kwargs...)
     OpenBoundaryCondition(val; kwargs...)     = BoundaryCondition(Open, val; kwargs...)
CommunicationBoundaryCondition(val; kwargs...) = BoundaryCondition(Communication, val; kwargs...)

# Support for various types of boundary conditions.
#
# Notes:
#     * "two-dimensional forms" (i, j, grid, ...) support boundary conditions on "native" grid boundaries:
#       east, west, etc.
#     * additional arguments to `fill_halo_regions` enter `getbc` after the `grid` argument:
#           so `fill_halo_regions!(c, clock, fields)` translates to `getbc(bc, i, j, grid, clock, fields)`, etc.

@inline getbc(bc, args...) = bc.condition(args...) # fallback!

@inline getbc(bc::BC{<:Open, Nothing}, i::Integer, j::Integer, grid::AbstractGrid, args...) = zero(eltype(grid))
@inline getbc(bc::BC{<:Flux, Nothing}, i::Integer, j::Integer, grid::AbstractGrid, args...) = zero(eltype(grid))
@inline getbc(bc::Nothing,             i::Integer, j::Integer, grid::AbstractGrid, args...) = zero(eltype(grid))

@inline getbc(bc::BC{C, <:Number}, args...) where C = bc.condition
@inline getbc(bc::BC{C, <:AbstractArray}, i::Integer, j::Integer, grid::AbstractGrid, args...) where C = @inbounds bc.condition[i, j]

Adapt.adapt_structure(to, bc::BoundaryCondition) = BoundaryCondition(Adapt.adapt(to, bc.classification),
                                                                     Adapt.adapt(to, bc.condition))

#####
##### Validation with topology
#####

validate_boundary_condition_topology(bc::Union{PBC, CBC, Nothing}, topo::Grids.Periodic, side) = nothing
validate_boundary_condition_topology(bc, topo::Grids.Periodic, side) =
    throw(ArgumentError("Cannot set $side $bc in a `Periodic` direction!"))

validate_boundary_condition_topology(::Nothing, topo::Flat, side) = nothing
validate_boundary_condition_topology(bc, topo::Flat, side) =
    throw(ArgumentError("Cannot set $side $bc in a `Flat` direction!"))

validate_boundary_condition_topology(bc, topo, side) = nothing

#####
##### Validation with architecture
#####

validate_boundary_condition_architecture(bc, arch, side) = nothing

validate_boundary_condition_architecture(bc::BoundaryCondition, arch, side) =
    validate_boundary_condition_architecture(bc.condition, arch, bc, side)

validate_boundary_condition_architecture(condition, arch, bc, side) = nothing
validate_boundary_condition_architecture(::Array, ::CPU, bc, side) = nothing
validate_boundary_condition_architecture(::CuArray, ::CUDAGPU, bc, side) = nothing
validate_boundary_condition_architecture(::ROCArray, ::ROCMGPU, bc, side) = nothing

validate_boundary_condition_architecture(::CuArray, ::CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `CuArray` on CPU architectures!"))

validate_boundary_condition_architecture(::CuArray, ::ROCMGPU, bc, side) =
    throw(ArgumentError("$side $bc must use `ROCArray` rather than `CuArray` on ROCMGPU architectures!"))

validate_boundary_condition_architecture(::Array, ::CUDAGPU, bc, side) =
    throw(ArgumentError("$side $bc must use `CuArray` rather than `Array` on CUDAGPU architectures!"))

validate_boundary_condition_architecture(::Array, ::ROCMGPU, bc, side) =
    throw(ArgumentError("$side $bc must use `ROCArray` rather than `Array` on ROCMGPU architectures!"))

validate_boundary_condition_architecture(::ROCArray, ::CUDAGPU, bc, side) =
    throw(ArgumentError("$side $bc must use `CuArray` rather than `ROCArray` on CUDAGPU architectures!"))

validate_boundary_condition_architecture(::ROCArray, ::CPU, bc, side) =
    throw(ArgumentError("$side $bc must use `Array` rather than `ROCArray` on CPU architectures!"))
