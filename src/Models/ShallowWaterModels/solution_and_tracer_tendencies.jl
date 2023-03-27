using Oceananigans.Advection
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Oceananigans.Coriolis
using Oceananigans.Operators
using Oceananigans.TurbulenceClosures: ∇_dot_qᶜ, ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ

@inline half_g_h²(i, j, k, grid, h, g)  = @inbounds 1/2 * g * h[i, j, k]^2
@inline h_plus_hB(i, j, k, grid, h, hB) = @inbounds h[i, j, k] + hB[i, j, k]

@inline x_pressure_gradient(i, j, k, grid, g, gprime, h, hB, ::SingleLayerModel, ::ConservativeFormulation) = ∂xᶠᶜᶜ(i, j, k, grid, half_g_h², h, g)
@inline y_pressure_gradient(i, j, k, grid, g, gprime, h, hB, ::SingleLayerModel, ::ConservativeFormulation) = ∂yᶜᶠᶜ(i, j, k, grid, half_g_h², h, g)
    
@inline function x_pressure_gradient(i, j, k, grid, g, gprime, h, hB, ::MultiLayerModel, ::ConservativeFormulation)
    Nz = grid.Nz
    ## Should be set to grid dimensions
    η[i, j, k] = hB[i, j, k] + cumsum(h, dims = 3)[i, j, k]
    ## Should be set to grid dimensions
    pressure[i, j, k] = 0
    for iter in range(k, Nz, step = 1)
        pressure[i, j, k] = pressure[i, j, k] + gprime[iter] * η[i, j, iter]
    end
    pressure[i, j, k] = pressure[i, j, k] + g * η[i, j, Nz]
    return ∂xᶠᶜᶜ(i, j, k, grid, h[i, j, k] * pressure[i, j, k], h, g)
end

@inline function y_pressure_gradient(i, j, k, grid, g, gprime, h, hB, ::MultiLayerModel, ::ConservativeFormulation)
    
    Nz = grid.Nz
    ## Should be set to grid dimensions
    η[i, j, k] = hB[i, j, k] + cumsum(h, dims = 3)[i, j, k]
    ## Should be set to grid dimensions
    pressure[i, j, k] = 0
    for iter in range(k, Nz, step = 1)
        pressure[i, j, k] = pressure[i, j, k] + gprime[iter] * η[i, j, iter]
    end
    pressure[i, j, k] = pressure[i, j, k] + g * η[i, j, Nz]
    return ∂xᶠᶜᶜ(i, j, k, grid, h[i, j, k] * pressure[i, j, k], h, g)
end

@inline x_pressure_gradient(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = g * ∂xᶠᶜᶜ(i, j, k, grid, h_plus_hB, h, hB)
@inline y_pressure_gradient(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = g * ∂yᶜᶠᶜ(i, j, k, grid, h_plus_hB, h, hB)

@inline bathymetry_contribution_x(i, j, k, grid, g, h, hB, formulation) = g * h[i, j, k] * ∂xᶠᶜᶜ(i, j, k, grid, hB)
@inline bathymetry_contribution_y(i, j, k, grid, g, h, hB, formulation) = g * h[i, j, k] * ∂yᶜᶠᶜ(i, j, k, grid, hB)

@inline bathymetry_contribution_x(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = zero(grid) 
@inline bathymetry_contribution_y(i, j, k, grid, g, h, hB, ::VectorInvariantFormulation) = zero(grid) 

"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      reduced_gravity,
                                      advection,
                                      velocities,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

    g = gravitational_acceleration
    gprime = reduced_gravity
    vertical_formulation = grid.vertical_formulation

    model_fields = shallow_water_fields(velocities, tracers, solution, formulation)

    return ( - div_mom_u(i, j, k, grid, advection, solution, formulation)
             - x_pressure_gradient(i, j, k, grid, g, gprime, solution.h, bathymetry, vertical_formulation, formulation)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             - bathymetry_contribution_x(i, j, k, grid, g, solution.h, bathymetry, vertical_formulation, formulation)
             - sw_∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, formulation)
             + forcings[1](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      reduced_gravity,
                                      advection,
                                      velocities,
                                      coriolis,
                                      closure,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock,
                                      formulation)

     g = gravitational_acceleration
     gprime = reduced_gravity
     vertical_formulation = grid.vertical_formulation

     model_fields = shallow_water_fields(velocities, tracers, solution, formulation)

    return ( - div_mom_v(i, j, k, grid, advection, solution, formulation)
             - y_pressure_gradient(i, j, k, grid, g, gprime, solution.h, bathymetry, formulation)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             - bathymetry_contribution_y(i, j, k, grid, g, solution.h, bathymetry, formulation)
             - sw_∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, diffusivities, clock, model_fields, formulation)
             + forcings[2](i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     advection,
                                     coriolis,
                                     closure,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock,
                                     formulation)

    return ( - div_Uh(i, j, k, grid, advection, solution, formulation)
             + forcings.h(i, j, k, grid, clock, merge(solution, tracers)))
end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 closure,
                                 solution,
                                 tracers,
                                 diffusivities,
                                 forcing,
                                 clock,
                                 formulation) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, solution, c, formulation) 
             + c_div_U(i, j, k, grid, solution, c, formulation)         
             + forcing(i, j, k, grid, clock, merge(solution, tracers)) 
            )
end
