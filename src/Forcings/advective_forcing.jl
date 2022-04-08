using Oceananigans.Advection: UpwindBiasedFifthOrder, div_Uc, div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.Fields: ZeroField, ConstantField

maybe_constant_field(u) = u
maybe_constant_field(u::Number) = ConstantField(u)

struct AdvectiveForcing{U, S, F, C}
    velocities :: U
    advection_scheme :: S
    advection_kernel_function :: F
    advected_field :: C
end

"""
AdvectiveForcing(scheme=UpwindBiasedFifthOrder(), u=ZeroField(), v=ZeroField(), w=ZeroField())

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

w_Stokes = - 2/9 * Δb / ν * R^2

settling = AdvectiveForcing(UpwindBiasedFifthOrder(), w=w_Stokes)

# output
AdvectiveForcing with the UpwindBiasedFifthOrder scheme:
├── u: ZeroField{Int64}
├── v: ZeroField{Int64}
└── w: -1.97096
```
"""
function AdvectiveForcing(scheme=UpwindBiasedFifthOrder(); u=ZeroField(), v=ZeroField(), w=ZeroField())
    u, v, w = maybe_constant_field.((u, v, w))
    velocities = (; u, v, w)
    return AdvectiveForcing(velocities, scheme, nothing, nothing) # stub
end

function regularize_forcing(af::AdvectiveForcing, field, field_name, model_field_names)
    kernel_function = field_name === :u ? div_𝐯u :
                      field_name === :v ? div_𝐯v :
                      field_name === :w ? div_𝐯w : div_Uc
    
    return AdvectiveForcing(af.velocities, af.advection_scheme, kernel_function, field)
end

@inline (af::AdvectiveForcing)(i, j, k, grid, clock, model_fields) =
    - af.advection_kernel_function(i, j, k, grid, af.advection_scheme, af.velocities, af.advected_field)

Base.summary(af::AdvectiveForcing) = string("AdvectiveForcing with the ", nameof(typeof(af.advection_scheme)), " scheme")

function Base.show(io::IO, af::AdvectiveForcing)
    
    print(io, summary(af), ":", '\n')

    print(io, "├── u: ", prettysummary(af.velocities.u), '\n',
              "├── v: ", prettysummary(af.velocities.v), '\n',
              "└── w: ", prettysummary(af.velocities.w))
end

