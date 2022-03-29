using Oceananigans.Advection: UpwindBiasedFifthOrder,div_Uc
using Oceananigans.Fields: ZeroField

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

settling = AdvectiveForcing(WENO5(), w=w_Stokes)
"""
function AdvectiveForcing(scheme=UpwindBiasedFifthOrder(); u=ZeroField(), v=ZeroField(), w=ZeroField())
    velocities = (; u, v, w)
    return AdvectiveForcing(velocities, scheme, nothing, nothing) # stub
end
    
function regularize_forcing(af::AdvectiveForcing, field, field_name, model_field_names)
    kernel_function =
        field_name === :u ? div_𝐯u :
        field_name === :v ? div_𝐯v :
        field_name === :w ? div_𝐯w : div_Uc

    return AdvectiveForcing(af.velocities, af.advection_scheme, kernel_function, field)
end

# div_Uc(i, j, k, grid, advection, velocities, c)

@inline (forcing::AdvectiveForcing)(i, j, k, grid, clock, model_fields) =
    af.advection_kernel_function(i, j, k, grid, af.advection_scheme, af.velocities, af.field)

