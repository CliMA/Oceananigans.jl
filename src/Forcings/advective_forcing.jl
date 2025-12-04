using Oceananigans.Fields: ZeroField
using Oceananigans.Utils: sum_of_velocities
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition, mask_immersed_field!
using Adapt

maybe_field(u::AbstractField, args...) = u

function maybe_field(u::Number, loc, grid, u_bcs, open_boundaries)
    immersed_bcs = ImmersedBoundaryCondition(; u_bcs...)
    bcs = FieldBoundaryConditions(grid, loc; u_bcs..., immersed = immersed_bcs)
    u_field = Field(loc, grid; boundary_conditions = bcs)
    set!(u_field, u)
    fill_halo_regions!(u_field)
    if open_boundaries
        mask_immersed_field!(u_field, u)
    else
        mask_immersed_field!(u_field, zero(eltype(grid)))
    end
    return u_field
end

maybe_field(u, args...) = throw(ArgumentError("For now only `Field`s and `Number`s are supported in AdvectiveForcing."))

struct AdvectiveForcing{U, V, W}
    u :: U
    v :: V
    w :: W
end

"""
    AdvectiveForcing(; grid=nothing, u=ZeroField(), v=ZeroField(), w=ZeroField(), open_boundaries=false)

Build a forcing term representing advection by the velocity field `u, v, w`.

# Keyword Arguments
- `grid`: Required when `u`, `v`, or `w` are numbers rather than fields
- `u`, `v`, `w`: Velocity components (can be numbers, fields, or functions)
- `open_boundaries`: If `true`, uses `OpenBoundaryCondition(velocity)` for boundary conditions.
                     If `false` (default), uses `OpenBoundaryCondition(nothing)`.

Example
=======

# Using a tracer field to model sinking particles

```jldoctest
using Oceananigans

grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1))

# Physical parameters
gravitational_acceleration          = 9.81     # m s⁻²
ocean_density                       = 1026     # kg m⁻³
mean_particle_density               = 2000     # kg m⁻³
mean_particle_radius                = 1e-3     # m
ocean_molecular_kinematic_viscosity = 1.05e-6  # m² s⁻¹

# Terminal velocity of a sphere in viscous flow
Δb = gravitational_acceleration * (mean_particle_density - ocean_density) / ocean_density
ν = ocean_molecular_kinematic_viscosity
R = mean_particle_radius

w_Stokes = - 2/9 * Δb / ν * R^2 # m s⁻¹

settling = AdvectiveForcing(w=w_Stokes; grid)

# output
AdvectiveForcing:
├── u: ZeroField{Int64}
├── v: ZeroField{Int64}
└── w: 1×1×2 Field{Center, Center, Face} on RectilinearGrid on CPU
```
"""
function AdvectiveForcing(; grid=nothing, u=ZeroField(), v=ZeroField(), w=ZeroField(), open_boundaries=false)
    if any((isa(u, Number), isa(v, Number), isa(w, Number))) && grid === nothing
        throw(ArgumentError("If passing numbers for u, v, w, you must also pass a grid"))
    end

    if open_boundaries
        u_bc = OpenBoundaryCondition(u)
        v_bc = OpenBoundaryCondition(v)
        w_bc = OpenBoundaryCondition(w)
    else
        u_bc = OpenBoundaryCondition(nothing)
        v_bc = OpenBoundaryCondition(nothing)
        w_bc = OpenBoundaryCondition(nothing)
    end

    u_bcs = (; east = u_bc, west = u_bc)
    v_bcs = (; south = v_bc, north = v_bc)
    w_bcs = (; bottom = w_bc, top = w_bc)

    u = maybe_field(u, (Face(), Center(), Center()), grid, u_bcs, open_boundaries)
    v = maybe_field(v, (Center(), Face(), Center()), grid, v_bcs, open_boundaries)
    w = maybe_field(w, (Center(), Center(), Face()), grid, w_bcs, open_boundaries)

    return AdvectiveForcing(u, v, w)
end

@inline (af::AdvectiveForcing)(i, j, k, grid, clock, model_fields) = 0

Base.summary(::AdvectiveForcing) = string("AdvectiveForcing")

function Base.show(io::IO, af::AdvectiveForcing)

    print(io, summary(af), ":", "\n")

    print(io, "├── u: ", prettysummary(af.u), "\n",
              "├── v: ", prettysummary(af.v), "\n",
              "└── w: ", prettysummary(af.w))
end

Adapt.adapt_structure(to, af::AdvectiveForcing) =
    AdvectiveForcing(adapt(to, af.u), adapt(to, af.v), adapt(to, af.w))

on_architecture(to, af::AdvectiveForcing) =
    AdvectiveForcing(on_architecture(to, af.u), on_architecture(to, af.v), on_architecture(to, af.w))

@inline velocities(forcing::AdvectiveForcing) = (u=forcing.u, v=forcing.v, w=forcing.w)

# fallback
@inline with_advective_forcing(forcing, total_velocities) = total_velocities

@inline with_advective_forcing(forcing::AdvectiveForcing, total_velocities) =
    sum_of_velocities(velocities(forcing), total_velocities)

# Unwrap the tuple within MultipleForcings
@inline with_advective_forcing(mf::MultipleForcings, total_velocities) =
    with_advective_forcing(mf.forcings, total_velocities)

# Recurse over forcing tuples
@inline with_advective_forcing(forcing::Tuple, total_velocities) =
    @inbounds with_advective_forcing(forcing[2:end], with_advective_forcing(forcing[1], total_velocities))

# Terminate recursion
@inline with_advective_forcing(forcing::NTuple{1}, total_velocities) =
    @inbounds with_advective_forcing(forcing[1], total_velocities)
