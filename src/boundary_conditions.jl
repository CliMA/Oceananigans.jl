#####
##### The basic idea: boundary condition types and data.
#####

abstract type BCType end
struct Periodic <: BCType end
struct Flux <: BCType end
struct Gradient <: BCType end
struct Value <: BCType end
struct NoPenetration <: BCType end

# Famous people.
const Dirchlet = Value
const Neumann = Gradient

"""
    BoundaryCondition(BCType, condition)

Construct a boundary condition of `BCType` with `condition`,
where `BCType` is `Flux` or `Gradient`. `condition` may be a
number, array, or a function with signature:

    `condition(i, j, grid, t, iter, U, Φ) = # function definition`

where `i` and `j` are indices along the boundary.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

# Some abbreviations to make our life easier.
const BC = BoundaryCondition
const FBC = BoundaryCondition{<:Flux}
const PBC = BoundaryCondition{<:Periodic}
const NPBC = BoundaryCondition{<:NoPenetration}
const VBC = BoundaryCondition{<:Value}
const GBC = BoundaryCondition{<:Gradient}
const NFBC = BoundaryCondition{Flux, Nothing}

BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)
bctype(bc::BoundaryCondition{C}) where C = C

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BC{C, A}) where {C<:BCType, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

PeriodicBC() = BoundaryCondition(Periodic, nothing)
NoPenetrationBC() = BoundaryCondition(NoPenetration, nothing)
NoFluxBC() = BoundaryCondition(Flux, nothing)


#####
##### Boundary conditions along particular coordinates
#####

"""
    CoordinateBoundaryConditions(left, right)

Construct `CoordinateBoundaryConditions` to be applied along coordinate `c`, where
`c` is `:x`, `:y`, or `:z`. A CoordinateBoundaryCondition has two fields
`left` and `right` that store boundary conditions on the 'left' (negative side)
and 'right' (positive side) of a given coordinate.
"""
mutable struct CoordinateBoundaryConditions{L, R}
     left :: L
    right :: R
end

const CBC = CoordinateBoundaryConditions
PeriodicBCs() = CBC(PeriodicBC(), PeriodicBC())

#=
Here we overload setproperty! and getproperty to permit users to call
the 'right' and 'left' bcs in the z-direction 'bottom' and 'top'.

Note that 'right' technically corresponds to face point N+1. Thus
the fact that right == bottom is associated with the reverse z-indexing
convention. With ordinary indexing, right == top.
=#
Base.setproperty!(cbc::CBC, side::Symbol, bc) = setbc!(cbc, Val(side), bc)
setbc!(cbc::CBC, ::Val{S}, bc) where S = setfield!(cbc, S, bc)
setbc!(cbc::CBC, ::Val{:bottom}, bc) = setfield!(cbc, :right, bc)
setbc!(cbc::CBC, ::Val{:top}, bc) = setfield!(cbc, :left, bc)

Base.getproperty(cbc::CBC, side::Symbol) = getbc(cbc, Val(side))
getbc(cbc::CBC, ::Val{S}) where S = getfield(cbc, S)
getbc(cbc::CBC, ::Val{:bottom}) = getfield(cbc, :right)
getbc(cbc::CBC, ::Val{:top}) = getfield(cbc, :left)

#####
##### Boundary conditions for Fields
#####

"""
    FieldBoundaryConditions(x, y, z)

Construct `FieldBoundaryConditions` for a field.
A FieldBoundaryCondition has `CoordinateBoundaryConditions` in
`x`, `y`, and `z`.
"""
FieldBoundaryConditions(x, y, z) = (x=x, y=y, z=z)

function FieldBoundaryConditions(;
    x = CoordinateBoundaryConditions(),
    y = CoordinateBoundaryConditions(),
    z = CoordinateBoundaryConditions()
    )
    return FieldBoundaryConditions(x, y, z)
end

"""
    HorizontallyPeriodicBCs(   top = BoundaryCondition(Flux, 0),
                            bottom = BoundaryCondition(Flux, 0))

Construct horizontally-periodic boundary conditions for ``u``, ``v``, or a
tracer field with top boundary condition (positive-z) `top`
and bottom boundary condition (negative-z) `bottom`.
"""
function HorizontallyPeriodicBCs(;    top = BoundaryCondition(Flux, 0),
                                   bottom = BoundaryCondition(Flux, 0))

    x = PeriodicBCs()
    y = PeriodicBCs()
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

"""
    ChannelBCs(;  north = BoundaryCondition(Flux, 0),
                  south = BoundaryCondition(Flux, 0),
                    top = BoundaryCondition(Flux, 0),
                 bottom = BoundaryCondition(Flux, 0))

Construct 'channel' boundary conditions (periodic in ``x``, non-periodic in
``y`` and ``z``) for ``u`` or a tracer field. The keywords `north`, `south`,
`top` and `bottom` correspond to boundary conditions in the positive ``y``,
negative ``y``, positive ``z`, and negative ``z`` directions respectively.
"""
function ChannelBCs(;  north = BoundaryCondition(Flux, 0),
                       south = BoundaryCondition(Flux, 0),
                         top = BoundaryCondition(Flux, 0),
                      bottom = BoundaryCondition(Flux, 0)
                    )

    x = PeriodicBCs()
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

#####
##### Boundary conditions for entire models
#####

"""
    ModelBoundaryConditions(u, v, w, T, S)

Construct a NamedTuple of boundary conditions for a model
with solution fields `u`, `v`, `w`, `T`, and `S`.
"""
ModelBoundaryConditions(u, v, w, T, S) = (u=u, v=v, w=w, T=T, S=S)

"""
    HorizontallyPeriodicModelBCs(u=HorizontallyPeriodicBCs, ...)

Construct a NamedTuple of boundary conditions for a horizontally-periodic
model with solution fields `u`, `v`, `w`, `T`, and `S`. Non-default
boundary conditions for any field must be horizontally-periodic.
"""
function HorizontallyPeriodicModelBCs(;
    u = HorizontallyPeriodicBCs(),
    v = HorizontallyPeriodicBCs(),
    w = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    T = HorizontallyPeriodicBCs(),
    S = HorizontallyPeriodicBCs()
   )
    return ModelBoundaryConditions(u, v, w, T, S)
end

function ChannelModelBCs(;
    u = ChannelBCs(),
    v = ChannelBCs(north=NoPenetrationBC(), south=NoPenetrationBC()),
    w = ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    T = ChannelBCs(),
    S = ChannelBCs()
   )
    return ModelBoundaryConditions(u, v, w, T, S)
end

# Default
const BoundaryConditions = HorizontallyPeriodicModelBCs

#####
##### Tendency and pressure boundary condition "translators":
#####
#####   * Boundary conditions on tendency terms
#####     derived from the boundary conditions on their repsective fields.
#####
#####   * Boundary conditions on pressure are derived from boundary conditions
#####     on the north-south horizontal velocity, v.
#####

TendencyBC(::BC) = BoundaryCondition(Flux, 0)
TendencyBC(::PBC) = PeriodicBC()
TendencyBC(::NPBC) = NoPenetrationBC()

TendencyCoordinateBCs(bcs) = CoordinateBoundaryConditions(TendencyBC(bcs.left), TendencyBC(bcs.right))

TendencyFieldBoundaryConditions(fieldbcs) = 
    NamedTuple{(:x, :y, :z)}(Tuple(TendencyCoordinateBCs(bcs) for bcs in fieldbcs))

TendenciesBoundaryConditions(modelbcs) =
    NamedTuple{propertynames(modelbcs)}(Tuple(TendencyFieldBoundaryConditions(bcs) for bcs in modelbcs))

# Pressure boundary conditions are either zero flux (Neumann) or Periodic.
# Note that a zero flux boundary condition is simpler than a zero gradient boundary condition.
PressureBC(::BC) = BoundaryCondition(Flux, 0)
PressureBC(::PBC) = PeriodicBC()

function PressureBoundaryConditions(vbcs)
    x = CoordinateBoundaryConditions(PressureBC(vbcs.x.left), PressureBC(vbcs.x.right))
    y = CoordinateBoundaryConditions(PressureBC(vbcs.y.left), PressureBC(vbcs.y.right))
    z = CoordinateBoundaryConditions(PressureBC(vbcs.z.left), PressureBC(vbcs.z.right))
    return (x=x, y=y, z=z)
end

#####
##### Algorithm for adding fluxes associated with non-trivial flux boundary conditions.
##### Inhomogeneous Value and Gradient boundary conditions are handled by filling halos.
#####

# Avoid some computation / memory accesses for Value, Gradient, Periodic, NoPenetration,
# and no-flux boundary conditions --- every boundary condition that does *not* prescribe
# a non-trivial flux.
const NotFluxBC = Union{VBC, GBC, PBC, NPBC, NFBC}
apply_z_bcs!(Gc, arch, grid, ::NotFluxBC, ::NotFluxBC, args...) = nothing

"""
    apply_z_bcs!(Gc, arch, grid, top_bc, bottom_bc, t, iter, U, Φ)

Apply flux boundary conditions to `c` by adding the associated flux divergence to the 
source term `Gc`.
"""
function apply_z_bcs!(Gc, arch, grid, top_bc, bottom_bc, args...)
    @launch device(arch) config=launch_config(grid, 2) _apply_z_bcs!(Gc, grid, top_bc, bottom_bc, args...)
    return
end

# Multiple dispatch on the type of boundary condition
getbc(bc::BC{C, <:Number}, args...)              where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, i, j, args...) where C = bc.condition[i, j]
getbc(bc::BC{C, <:Function}, args...)            where C = bc.condition(args...)

Base.getindex(bc::BC{C, <:AbstractArray}, inds...) where C = getindex(bc.condition, inds...)

# Fall back functions for non-Flux boundary conditions.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

# Shortcuts for 'no-flux' boundary conditions.
@inline apply_z_top_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_z_bottom_bc!(Gc, ::NFBC, args...) = nothing

"""
    apply_z_top_bc!(Gc, top_bc, i, j, grid, t, iter, U, Φ)

Add the part of flux divergence associated with a top boundary condition on c.
Note that because

        tendency = ∂c/∂t = Gc = - ∇ ⋅ flux

A positive top flux is associated with a *decrease* in `Gc` near the top boundary.
If `top_bc.condition` is a function, the function must have the signature

    `top_bc.condition(i, j, grid, t, iter, U, Φ)`

"""
@inline apply_z_top_bc!(Gc, top_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, 1] -= getbc(top_flux, i, j, grid, args...) / grid.Δz

"""
    apply_z_bottom_bc!(Gc, bottom_bc, i, j, grid, t, iter, U, Φ)

Add the flux divergence associated with a bottom flux boundary condition on c.
Note that because

        tendency = ∂c/∂t = Gc = - ∇ ⋅ flux

A positive bottom flux is associated with an *increase* in `Gc` near the bottom boundary.
If `bottom_bc.condition` is a function, the function must have the signature

    `bottom_bc.condition(i, j, grid, t, iter, U, Φ)`

"""
@inline apply_z_bottom_bc!(Gc, bottom_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, grid.Nz] += getbc(bottom_flux, i, j, grid, args...) / grid.Δz

"""
    apply_z_bcs!(top_bc, bottom_bc, grid, c, Gc, κ, closure, t, iter, U, Φ)

Apply a top and/or bottom boundary condition to variable c.
"""
function _apply_z_bcs!(Gc, grid, top_bc, bottom_bc, args...)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
               apply_z_top_bc!(Gc, top_bc,    i, j, grid, args...) 
            apply_z_bottom_bc!(Gc, bottom_bc, i, j, grid, args...) 
        end
    end
end
