#####
##### The basic idea: boundary condition types and data.
#####

"""
    BCType

Abstract supertype for boundary condition types.
"""
abstract type BCType end

"""
    Periodic

A type specifying a periodic boundary condition.

A condition may not be specified with a `Periodic` boundary condition.
"""
struct Periodic <: BCType end

"""
    Flux

A type specifying a boundary condition on the flux of a field.
"""
struct Flux <: BCType end

"""
    Gradient

A type specifying a boundary condition on the derivative or gradient of a field. Also
called a Neumann boundary condition.
"""
struct Gradient <: BCType end

"""
    Value

A type specifying a boundary condition on the value of a field. Also called a Dirchlet
boundary condition.
"""
struct Value <: BCType end

"""
    NoPenetration

A type specifying a no-penetration boundary condition for a velocity component that is normal to a wall.

Thus `NoPenetration` can only be applied to `u` along x, `v` along y, or `w` along z. For all other cases --- fields
located at (Cell, Cell, Cell), or `u`, `v`, and `w` in (y, z), (x, z), and (x, y), respectively, either `Value`,
`Gradient`, or `Flux` conditions must be used.

A condition may not be specified with a `NoPenetration` boundary condition.

Note that this differs from a zero `Value` boundary condition as `Value` imposes values at the cell centers
(and could apply to tracers) while a no-penetration boundary condition only applies
to normal velocity components at a wall, where the velocity at the cell face collocated
at the wall is known and set to zero.
"""
struct NoPenetration <: BCType end

# Famous people

"""
    Dirchlet

An alias for the `Value` boundary condition type.
"""
const Dirchlet = Value

"""
    Neumann

An alias for the `Gradient` boundary condition type.
"""
const Neumann = Gradient

"""
    BoundaryCondition{C<:BCType}(condition)

Construct a boundary condition of type `C` with a `condition` that may be given by a
number, an array, or a function with signature:

    `condition(i, j, grid, time, iteration, U, Φ, parameters) = # function definition`

that returns a number and where `i` and `j` are indices along the boundary.

Boundary condition types include `Periodic`, `Flux`, `Value`, `Gradient`, and `NoPenetration`.
"""
struct BoundaryCondition{C<:BCType, T}
    condition :: T
end

# Some abbreviations to make life easier.
const BC   = BoundaryCondition
const FBC  = BoundaryCondition{<:Flux}
const PBC  = BoundaryCondition{<:Periodic}
const NPBC = BoundaryCondition{<:NoPenetration}
const VBC  = BoundaryCondition{<:Value}
const GBC  = BoundaryCondition{<:Gradient}
const NFBC = BoundaryCondition{Flux, Nothing}

BoundaryCondition(Tbc, c) = BoundaryCondition{Tbc, typeof(c)}(c)
bctype(bc::BoundaryCondition{C}) where C = C

# Adapt boundary condition struct to be GPU friendly and passable to GPU kernels.
Adapt.adapt_structure(to, b::BC{C, A}) where {C<:BCType, A<:AbstractArray} =
    BoundaryCondition(C, Adapt.adapt(to, parent(b.condition)))

PeriodicBC() = BoundaryCondition(Periodic, nothing)
NoPenetrationBC() = BoundaryCondition(NoPenetration, nothing)
NoFluxBC() = BoundaryCondition(Flux, nothing)

# Multiple dispatch on the type of boundary condition
getbc(bc::BC{C, <:Number}, args...)              where C = bc.condition
getbc(bc::BC{C, <:AbstractArray}, i, j, args...) where C = bc.condition[i, j]
getbc(bc::BC{C, <:Function}, args...)            where C = bc.condition(args...)

Base.getindex(bc::BC{C, <:AbstractArray}, inds...) where C = getindex(bc.condition, inds...)

#####
##### Boundary conditions along particular coordinates
#####

"""
    CoordinateBoundaryConditions(left, right)

A set of two `BoundaryCondition`s to be applied along a coordinate x, y, or z.

The `left` boundary condition is applied on the negative or lower side of the coordinate
while the `right` boundary condition is applied on the positive or higher side.
"""
mutable struct CoordinateBoundaryConditions{L, R}
     left :: L
    right :: R
end

const CBC = CoordinateBoundaryConditions
PeriodicBCs() = CBC(PeriodicBC(), PeriodicBC())

# Here we overload setproperty! and getproperty to permit users to call
# the 'right' and 'left' bcs in the z-direction 'bottom' and 'top'.
#
# Note that 'right' technically corresponds to face point N+1. Thus
# the fact that right == bottom is associated with the reverse z-indexing
# convention. With ordinary indexing, right == top.
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
    FieldBoundaryConditions

An alias for `NamedTuple{(:x, :y, :z)}` that represents a set of three `CoordinateBoundaryCondition`s
applied to a field along x, y, and z.
"""
const FieldBoundaryConditions = NamedTuple{(:x, :y, :z)}

"""
    FieldBoundaryConditions(x, y, z)

Construct a `FieldBoundaryConditions` using a `CoordinateBoundaryCondition` for each of the
`x`, `y`, and `z` coordinates.
"""
FieldBoundaryConditions(x, y, z) = FieldBoundaryConditions((x, y, z))

"""
    HorizontallyPeriodicBCs(;   top = BoundaryCondition(Flux, nothing),
                             bottom = BoundaryCondition(Flux, nothing))

Construct `FieldBoundaryConditions` with `Periodic` boundary conditions in the x and y
directions and specified `top` (+z) and `bottom` (-z) boundary conditions for u, v,
and tracer fields.

`HorizontallyPeriodicBCs` cannot be applied to the the vertical velocity w.
"""
function HorizontallyPeriodicBCs(;    top = BoundaryCondition(Flux, nothing),
                                   bottom = BoundaryCondition(Flux, nothing))

    x = PeriodicBCs()
    y = PeriodicBCs()
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

"""
    ChannelBCs(; north = BoundaryCondition(Flux, nothing),
                 south = BoundaryCondition(Flux, nothing),
                   top = BoundaryCondition(Flux, nothing),
                bottom = BoundaryCondition(Flux, nothing))

Construct `FieldBoundaryConditions` with `Periodic` boundary conditions in the x
direction and specified `north` (+y), `south` (-y), `top` (+z) and `bottom` (-z)
boundary conditions for u, v, and tracer fields.

`ChannelBCs` cannot be applied to the the vertical velocity w.
"""
function ChannelBCs(;  north = BoundaryCondition(Flux, nothing),
                       south = BoundaryCondition(Flux, nothing),
                         top = BoundaryCondition(Flux, nothing),
                      bottom = BoundaryCondition(Flux, nothing)
                    )

    x = PeriodicBCs()
    y = CoordinateBoundaryConditions(south, north)
    z = CoordinateBoundaryConditions(top, bottom)

    return FieldBoundaryConditions(x, y, z)
end

#####
##### Boundary conditions for model solutions
#####

"""
    SolutionBoundaryConditions(u, v, w, T, S)

Construct a `NamedTuple` of `FieldBoundaryConditions` for a model with solution fields
`u`, `v`, `w`, `T`, and `S`.
"""
SolutionBoundaryConditions(u, v, w, T, S) = (u=u, v=v, w=w, T=T, S=S)

"""
    HorizontallyPeriodicSolutionBCs(u=HorizontallyPeriodicBCs(), ...)

Construct `SolutionBoundaryConditions` for a horizontally-periodic model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `HorizontallyPeriodicBCs` are applied to `u`, `v`, `T`, and `S`
and `HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `HorizontallyPeriodicBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function HorizontallyPeriodicSolutionBCs(;
    u = HorizontallyPeriodicBCs(),
    v = HorizontallyPeriodicBCs(),
    w = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    T = HorizontallyPeriodicBCs(),
    S = HorizontallyPeriodicBCs()
   )
    return SolutionBoundaryConditions(u, v, w, T, S)
end

"""
    ChannelSolutionBCs(u=ChannelBCs(), ...)

Construct `SolutionBoundaryConditions` for a reentrant channel model
configuration with solution fields `u`, `v`, `w`, `T`, and `S` specified by keyword arguments.

By default `ChannelBCs` are applied to `u`, `v`, `T`, and `S`
and `ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())` is applied to `w`.

Use `ChannelBCs` when constructing non-default boundary conditions for `u`, `v`, `w`, `T`, `S`.
"""
function ChannelSolutionBCs(;
    u = ChannelBCs(),
    v = ChannelBCs(north=NoPenetrationBC(), south=NoPenetrationBC()),
    w = ChannelBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC()),
    T = ChannelBCs(),
    S = ChannelBCs()
   )
    return SolutionBoundaryConditions(u, v, w, T, S)
end

# Default semantics
const BoundaryConditions = HorizontallyPeriodicSolutionBCs

#####
##### Tendency and pressure boundary condition "translators":
#####
#####   * Boundary conditions on tendency terms are
#####     derived from the boundary conditions on their repsective fields.
#####
#####   * Boundary conditions on pressure are derived from boundary conditions
#####     on the north-south horizontal velocity, v.
#####

TendencyBC(::BC) = BoundaryCondition(Flux, nothing)
TendencyBC(::PBC) = PeriodicBC()
TendencyBC(::NPBC) = NoPenetrationBC()

TendencyCoordinateBCs(bcs) = CoordinateBoundaryConditions(TendencyBC(bcs.left), TendencyBC(bcs.right))

TendencyFieldBoundaryConditions(field_bcs) =
    FieldBoundaryConditions(Tuple(TendencyCoordinateBCs(bcs) for bcs in field_bcs))

TendenciesBoundaryConditions(solution_bcs) =
    NamedTuple{propertynames(solution_bcs)}(Tuple(TendencyFieldBoundaryConditions(bcs) for bcs in solution_bcs))

# Pressure boundary conditions are either zero flux (Neumann) or Periodic.
# Note that a zero flux boundary condition is simpler than a zero gradient boundary condition.
PressureBC(::BC) = BoundaryCondition(Flux, nothing)
PressureBC(::PBC) = PeriodicBC()

function PressureBoundaryConditions(vbcs)
    x = CoordinateBoundaryConditions(PressureBC(vbcs.x.left), PressureBC(vbcs.x.right))
    y = CoordinateBoundaryConditions(PressureBC(vbcs.y.left), PressureBC(vbcs.y.right))
    z = CoordinateBoundaryConditions(PressureBC(vbcs.z.left), PressureBC(vbcs.z.right))
    return (x=x, y=y, z=z)
end

#####
##### Model boundary conditions for pressure, and solution, and tendency solutions
#####

const ModelBoundaryConditions = NamedTuple{(:solution, :tendency, :pressure)}

function ModelBoundaryConditions(solution_boundary_conditions::NamedTuple)
    model_boundary_conditions = (solution = solution_boundary_conditions,
                                 tendency = TendenciesBoundaryConditions(solution_boundary_conditions),
                                 pressure = PressureBoundaryConditions(solution_boundary_conditions.v))
    return model_boundary_conditions
end

ModelBoundaryConditions(model_boundary_conditions::ModelBoundaryConditions) =
    model_boundary_conditions

#####
##### Algorithm for adding fluxes associated with non-trivial flux boundary conditions.
##### Inhomogeneous Value and Gradient boundary conditions are handled by filling halos.
#####

# Avoid some computation / memory accesses for Value, Gradient, Periodic, NoPenetration,
# and no-flux boundary conditions --- every boundary condition that does *not* prescribe
# a non-trivial flux.
const NotFluxBC = Union{VBC, GBC, PBC, NPBC, NFBC}
apply_z_bcs!(Gc, arch, grid, ::NotFluxBC, ::NotFluxBC, args...) = nothing
apply_y_bcs!(Gc, arch, grid, ::NotFluxBC, ::NotFluxBC, args...) = nothing
apply_x_bcs!(Gc, arch, grid, ::NotFluxBC, ::NotFluxBC, args...) = nothing

"""
    apply_z_bcs!(Gc, arch, grid, top_bc, bottom_bc, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the top and bottom.
"""
function apply_z_bcs!(Gc, arch, grid, top_bc, bottom_bc, args...)
    @launch device(arch) config=launch_config(grid, :xy) _apply_z_bcs!(Gc, grid, top_bc, bottom_bc, args...)
    return
end

function apply_y_bcs!(Gc, arch, grid, left_bc, right_bc, args...)
    @launch device(arch) config=launch_config(grid, :xz) _apply_y_bcs!(Gc, grid, left_bc, right_bc, args...)
    return
end

function apply_x_bcs!(Gc, arch, grid, left_bc, right_bc, args...)
    @launch device(arch) config=launch_config(grid, :yz) _apply_x_bcs!(Gc, grid, left_bc, right_bc, args...)
    return
end

"""
    _apply_z_bcs!(Gc, grid, top_bc, bottom_bc, args...)

Apply a top and/or bottom boundary condition to variable `c`.
"""
function _apply_z_bcs!(Gc, grid, top_bc, bottom_bc, args...)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
               apply_z_top_bc!(Gc, top_bc,    i, j, grid, args...)
            apply_z_bottom_bc!(Gc, bottom_bc, i, j, grid, args...)
        end
    end
end

function _apply_y_bcs!(Gc, grid, left_bc, right_bc, args...)
    @loop for k in (1:grid.Nz; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
             apply_y_left_bc!(Gc,  left_bc, i, k, grid, args...)
            apply_y_right_bc!(Gc, right_bc, i, k, grid, args...)
        end
    end
end

function _apply_x_bcs!(Gc, grid, left_bc, right_bc, args...)
    @loop for k in (1:grid.Nz; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for j in (1:grid.Ny; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
             apply_x_left_bc!(Gc, left_bc,  j, k, grid, args...)
            apply_x_right_bc!(Gc, right_bc, j, k, grid, args...)
        end
    end
end

# Fall back functions for boundary conditions that are not of type Flux.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

@inline apply_y_left_bc!(args...) = nothing
@inline apply_y_right_bc!(args...) = nothing

@inline apply_x_left_bc!(args...) = nothing
@inline apply_x_right_bc!(args...) = nothing

# Shortcuts for 'no-flux' boundary conditions.
@inline apply_z_top_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_z_bottom_bc!(Gc, ::NFBC, args...) = nothing

@inline apply_y_left_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_y_right_bc!(Gc, ::NFBC, args...) = nothing

@inline apply_x_left_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_x_right_bc!(Gc, ::NFBC, args...) = nothing

"""
    apply_z_top_bc!(Gc, top_flux::BC{<:Flux}, i, j, grid, args...)

Add the part of flux divergence associated with a top boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive top flux is associated with a *decrease* in `Gc` near the top boundary.
If `top_bc.condition` is a function, the function must have the signature

    `top_bc.condition(i, j, grid, boundary_condition_args...)`
"""
@inline apply_z_top_bc!(Gc, top_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, 1] -= getbc(top_flux, i, j, grid, args...) / grid.Δz

@inline apply_y_left_bc!(Gc, left_flux::BC{<:Flux}, i, k, grid, args...) =
    @inbounds Gc[i, 1, k] -= getbc(top_flux, i, k, grid, args...) / grid.Δy

@inline apply_x_left_bc!(Gc, left_flux::BC{<:Flux}, j, k, grid, args...) =
    @inbounds Gc[1, j, k] -= getbc(top_flux, j, k, grid, args...) / grid.Δx

"""
    apply_z_bottom_bc!(Gc, bottom_flux::BC{<:Flux}, i, j, grid, args...)

Add the flux divergence associated with a bottom flux boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive bottom flux is associated with an *increase* in `Gc` near the bottom boundary.
If `bottom_bc.condition` is a function, the function must have the signature

    `bottom_bc.condition(i, j, grid, boundary_condition_args...)`
"""
@inline apply_z_bottom_bc!(Gc, bottom_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, grid.Nz] += getbc(bottom_flux, i, j, grid, args...) / grid.Δz

@inline apply_y_right_bc!(Gc, right_flux::BC{<:Flux}, i, k, grid, args...) =
    @inbounds Gc[i, grid.Ny, k] += getbc(right_flux, i, k, grid, args...) / grid.Δy

@inline apply_x_right_bc!(Gc, right_flux::BC{<:Flux}, j, k, grid, args...) =
    @inbounds Gc[grid.Nx, j, k] += getbc(right_flux, j, k, grid, args...) / grid.Δx
