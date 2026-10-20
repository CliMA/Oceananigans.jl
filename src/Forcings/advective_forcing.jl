using Oceananigans.Fields: ZeroField, ConstantField
using Oceananigans.Fields: Field, compute!, fill_halo_regions!, instantiated_location
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, mask_immersed_normal_flow!
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Utils: sum_of_velocities

maybe_constant_field(u) = u
maybe_constant_field(u::Number) = ConstantField(u)

struct AdvectiveForcing{U, V, W}
    u :: U
    v :: V
    w :: W
end

"""
    AdvectiveForcing(; u=ZeroField(), v=ZeroField(), w=ZeroField())

Build a forcing term representing advection by the velocity field `u, v, w`.

Example
=======

# Using a tracer field to model sinking particles

```jldoctest
using Oceananigans

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

settling = AdvectiveForcing(w=w_Stokes)

# output
AdvectiveForcing:
├── u: ZeroField{Int64}
├── v: ZeroField{Int64}
└── w: ConstantField(-1.97096)
```
"""
function AdvectiveForcing(; u=ZeroField(), v=ZeroField(), w=ZeroField())
    u, v, w = maybe_constant_field.((u, v, w))
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

Architectures.on_architecture(to, af::AdvectiveForcing) =
    AdvectiveForcing(on_architecture(to, af.u), on_architecture(to, af.v), on_architecture(to, af.w))

@inline velocities(forcing::AdvectiveForcing) = (u=forcing.u, v=forcing.v, w=forcing.w)

has_field_advective_forcing(forcing::AdvectiveForcing) = any(velocity -> velocity isa Field, velocities(forcing))

function materialize_forcing(forcing::AdvectiveForcing, field, field_name, model_field_names)
    components = map(velocity -> materialize_advective_velocity(velocity, model_field_names), velocities(forcing))
    return AdvectiveForcing(components.u, components.v, components.w)
end

materialize_advective_velocity(velocity, model_field_names) = velocity

function materialize_advective_velocity(velocity::Field, model_field_names)
    grid = velocity.grid
    grid isa ImmersedBoundaryGrid || return velocity
    loc = instantiated_location(velocity)
    bcs = regularize_field_boundary_conditions(velocity.boundary_conditions, grid, loc, model_field_names)
    return Field(loc, grid, velocity.data, bcs, velocity.indices, velocity.operand, velocity.status)
end

function compute_forcing!(forcing::AdvectiveForcing, clock, model_fields)
    foreach(velocity -> refresh_advective_velocity!(velocity, clock, model_fields), velocities(forcing))
    return nothing
end

refresh_advective_velocity!(velocity, clock, model_fields) = nothing

function refresh_advective_velocity!(velocity::Field, clock, model_fields)
    isnothing(velocity.operand) || compute!(velocity)
    mask_immersed_normal_flow!(velocity, clock, model_fields)
    fill_halo_regions!(velocity, clock, model_fields)
    return nothing
end

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
