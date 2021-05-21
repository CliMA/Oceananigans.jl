import Adapt

"""
    struct BoundaryCondition{C<:BCType, T}

Container for boundary conditions.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

"""
    BoundaryCondition(BC, condition)

Construct a boundary condition of type `BC` with a number or array as a `condition`.

Boundary condition types include `Periodic`, `Flux`, `Value`, `Gradient`, and `NormalFlow`.
"""
BoundaryCondition(BC, condition) = BoundaryCondition{BC, typeof(condition)}(condition)

"""
    BoundaryCondition(BC, condition::Function; parameters=nothing, discrete_form=false)

Construct a boundary condition of type `BC` with a function boundary `condition`.

By default, the function boudnary `condition` is assumed to have the 'continuous form'
`condition(ξ, η, t)`, where `t` is time and `ξ` and `η` vary along the boundary.
In particular:

- On `x`-boundaries, `condition(y, z, t)`.
- On `y`-boundaries, `condition(x, z, t)`.
- On `z`-boundaries, `condition(x, y, t)`.

If `parameters` is not `nothing`, then function boundary conditions have the form
`func(ξ, η, t, parameters)`, where `ξ` and `η` are spatial coordinates varying along
the boundary as explained above.

If `discrete_form=true`, the function `condition` is assumed to have the "discrete form",

    `condition(i, j, grid, clock, model_fields)`,

where `i`, and `j` are indices that vary along the boundary. If `discrete_form=true` and
`parameters` is not `nothing`, the function `condition` is called with

    `condition(i, j, grid, clock, model_fields, parameters)`.
"""
function BoundaryCondition(TBC, condition::Function;
                           parameters = nothing,
                           discrete_form = false,
                           field_dependencies=())

    if discrete_form
        field_dependencies != () && error("Cannot set `field_dependencies` when `discrete_form=true`!")
        condition = DiscreteBoundaryFunction(condition, parameters)
    else
        # Note that the boundary :x and location Center, Center are in general incorrect.
        # These are corrected in the FieldBoundaryConditions constructor.
        condition = ContinuousBoundaryFunction(condition, parameters, field_dependencies)
    end

    return BoundaryCondition{TBC, typeof(condition)}(condition)
end

bctype(bc::BoundaryCondition{C}) where C = C

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BoundaryCondition{C, A}) where {C<:BCType, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

#####
##### Some abbreviations to make life easier.
#####

# These type aliases make dispatching on BCs easier (not exported).
const BC   = BoundaryCondition
const FBC  = BoundaryCondition{<:Flux}
const PBC  = BoundaryCondition{<:Periodic}
const NFBC = BoundaryCondition{<:NormalFlow}
const VBC  = BoundaryCondition{<:Value}
const GBC  = BoundaryCondition{<:Gradient}
const ZFBC = BoundaryCondition{Flux, Nothing} # "zero" flux

# More readable BC constructors for the public API.
    PeriodicBoundaryCondition() = BoundaryCondition(Periodic,   nothing)
      NoFluxBoundaryCondition() = BoundaryCondition(Flux,       nothing)
ImpenetrableBoundaryCondition() = BoundaryCondition(NormalFlow, nothing)

      FluxBoundaryCondition(val; kwargs...) = BoundaryCondition(Flux, val; kwargs...)
     ValueBoundaryCondition(val; kwargs...) = BoundaryCondition(Value, val; kwargs...)
  GradientBoundaryCondition(val; kwargs...) = BoundaryCondition(Gradient, val; kwargs...)
NormalFlowBoundaryCondition(val; kwargs...) = BoundaryCondition(NormalFlow, val; kwargs...)

# Support for various types of boundary conditions
@inline getbc(bc::BC{<:NormalFlow, Nothing}, i, j, grid, args...) = zero(eltype(grid))
@inline getbc(bc::BC{<:Flux, Nothing}, i, j, grid, args...) = zero(eltype(grid))

@inline getbc(bc::BC{C, <:Number},        args...)                         where C = bc.condition
@inline getbc(bc::BC{C, <:AbstractArray}, i, j, grid, clock, model_fields) where C = @inbounds bc.condition[i, j]
@inline getbc(bc::BC{C, <:Function},      i, j, grid, clock, model_fields) where C = bc.condition(i, j, grid, clock, model_fields)

@inline Base.getindex(bc::BC{C, <:AbstractArray}, i, j) where C = getindex(bc.condition, i, j)
