using Oceananigans.Advection: UpwindBiasedFifthOrder, div_Uc, div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.Fields: ZeroField, ConstantField
using Oceananigans.Utils: SumOfArrays
using Adapt

maybe_constant_field(u) = u
maybe_constant_field(u::Number) = ConstantField(u)

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

# fallback
@inline with_advective_forcing(forcing, total_velocities) = total_velocities

@inline with_advective_forcing(forcing::AdvectiveForcing, total_velocities) = 
    (u = SumOfArrays{2}(forcing.u, total_velocities.u),
     v = SumOfArrays{2}(forcing.v, total_velocities.v),
     w = SumOfArrays{2}(forcing.w, total_velocities.w))

@inline with_advective_forcing(forcing::Tuple, total_velocities) = 
    @inbounds with_advective_forcing(forcing[2:end], with_advective_forcing(forcing[1], total_velocities))

# terminates recursion
@inline with_advective_forcing(forcing::NTuple{1}, total_velocities) = 
    @inbounds with_advective_forcing(forcing[1], total_velocities)