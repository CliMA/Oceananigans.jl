using Oceananigans.Advection: UpwindBiasedFifthOrder
using Oceananigans.Fields: ZeroField

struct AdvectiveForcing{I, U, S, F}
    velocities :: U
    advection_scheme :: S
    advection_kernel_function :: F
end

"""
    AdvectiveForcing(scheme=UpwindBiasedFifthOrder(), u=ZeroField(), v=ZeroField(), w=ZeroField())

Build a forcing term representing advection by the velocity field `u, v, w` with an advection `scheme`.

Example
=======

# Using a tracer field to model sinking particles

```julia
# Physical parameters
gravitational_acceleration    = 9.81     # m s⁻²
ocean_density                 = 1026     # kg m⁻³
particle_density              = 2000     # kg m⁻³
molecular_kinematic_viscosity = 1.05e-6  # m² s⁻¹
mean_particle_radius          = 1e-3     # m

# Terminal velocity of a sphere in viscous flow
Δb = gravitational_acceleration * (particle_density - ocean_density)
μ = ocean_density * ocean_molecular_kinematic_viscosity
R = mean_particle_radius

w_Stokes = - 2/9 * Δb / μ * R^2

settling = AdvectiveForcing(WENO5(), w=w_Stokes)
"""
function AdvectiveForcing(scheme=UpwindBiasedFifthOrder(), u=ZeroField(), v=ZeroField(), w=ZeroField())
    velocities = (; u, v, w)
    return AdvectiveForcing{Nothing}(velocities, scheme, nothing) # stub
end
    
function regularize_forcing(af::AdvectiveForcing, field, field_name, model_field_names)
    kernel_function =
        field_name === :u ? div_𝐯u :
        field_name === :v ? div_𝐯v :
        field_name === :w ? div_𝐯w : div_Uc

    field_index = findfirst(name -> name === field_name, model_field_names) 

    return AdvectiveForcing{field_index}(af.velocities, af.advection_scheme, kernel_function)
end

# div_Uc(i, j, k, grid, advection, velocities, c)

@inline (forcing::AdvectiveForcing{I})(i, j, k, grid, clock, model_fields) where I =
    af.advection_kernel_function(i, j, k, grid, af.advection_scheme, af.velocities, model_fields[I])

