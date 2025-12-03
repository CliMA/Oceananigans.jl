using Oceananigans.Fields: ZeroField
using Oceananigans.Utils: sum_of_velocities
using Oceananigans.BoundaryConditions: OpenBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryCondition
using Adapt

maybe_field(u, args...) = u

function maybe_field(u::Number, loc, grid, u_bcs)
    immersed_bcs = ImmersedBoundaryCondition(; u_bcs...)
    bcs = FieldBoundaryConditions(grid, loc; u_bcs..., immersed = immersed_bcs)
    u_field = Field(loc, grid; boundary_conditions = bcs)
    set!(u_field, u)
    fill_halo_regions!(u_field)
    return u_field
end

struct AdvectiveForcing{U, V, W}
    u :: U
    v :: V
    w :: W
end

"""
    AdvectiveForcing(u=ZeroField(), v=ZeroField(), w=ZeroField())

Build a forcing term representing advection by the velocity field `u, v, w` with an advection `scheme`.

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
function AdvectiveForcing(; grid=nothing, u=ZeroField(), v=ZeroField(), w=ZeroField(), normal_boundary_condition=OpenBoundaryCondition(nothing))
    if any((isa(u, Number), isa(v, Number), isa(w, Number))) && grid === nothing
        throw(ArgumentError("If passing numbers for u, v, w, you must also pass a grid"))
    end

    u_bcs = (; east = normal_boundary_condition, west = normal_boundary_condition)
    v_bcs = (; south = normal_boundary_condition, north = normal_boundary_condition)
    w_bcs = (; bottom = normal_boundary_condition, top = normal_boundary_condition)

    u = maybe_field(u, (Face(), Center(), Center()), grid, u_bcs)
    v = maybe_field(v, (Center(), Face(), Center()), grid, v_bcs)
    w = maybe_field(w, (Center(), Center(), Face()), grid, w_bcs)

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
