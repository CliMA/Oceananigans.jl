using JULES.Operators

using Oceananigans: datatuple
using Oceananigans.BoundaryConditions

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Simulations: ab2_or_rk3_time_step!

#####
##### Utilities for time stepping
#####

function rk3_time_step(rk3_iter, Δt)
    rk3_iter == 1 && return Δt/3
    rk3_iter == 2 && return Δt/2
    rk3_iter == 3 && return Δt
end

#####
##### Time-stepping algorithm
#####

ab2_or_rk3_time_step!(model::CompressibleModel, Δt; euler) = time_step!(model, Δt)

function time_step!(model::CompressibleModel, Δt)
    total_density  = model.total_density
    momenta = model.momenta
    tracers = model.tracers
    diffusivities  = model.diffusivities
    slow_source_terms  = model.slow_source_terms
    fast_source_terms  = model.fast_source_terms
    intermediate_fields = model.intermediate_fields

    # On third RK3 step, we update Φ⁺ instead of model.intermediate_fields
    Φ⁺ = (momenta..., tracers = tracers)

    # On the first and second RK3 steps we want to update intermediate momenta and tracers.
    momenta_names = propertynames(momenta)
    tracers_names = propertynames(tracers)

    intermediate_momenta_fields = [getproperty(intermediate_fields, ρu)         for ρu in momenta_names]
    intermediate_tracers_fields = [getproperty(intermediate_fields.tracers, ρc) for ρc in tracers_names]

    intermediate_momenta = NamedTuple{momenta_names}(intermediate_momenta_fields)
    intermediate_tracers = NamedTuple{tracers_names}(intermediate_tracers_fields)

    @debug "Computing slow forcings..."
    update_total_density!(total_density, model.grid, model.gases, tracers)
    fill_halo_regions!(merge((Σρ=total_density,), momenta, tracers), model.architecture, model.clock, nothing)

    fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

    compute_slow_source_terms!(
        slow_source_terms, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.coriolis, model.closure, total_density, momenta, tracers, diffusivities, model.forcing, model.clock)

    fill_halo_regions!(slow_source_terms.ρw, model.architecture, model.clock, nothing)

    for rk3_iter in 1:3
        @debug "RK3 step #$rk3_iter..."
        @debug "  Computing right hand sides..."

        if rk3_iter == 1
            compute_rhs_args = (fast_source_terms, model.grid, model.thermodynamic_variable,
                                model.gases, model.gravity, total_density, momenta, tracers, slow_source_terms)

            update_total_density!(total_density, model.grid, model.gases, tracers)
            fill_halo_regions!(merge((Σρ=total_density,), momenta, tracers), model.architecture, model.clock, nothing)
        else
            compute_rhs_args = (fast_source_terms, model.grid, model.thermodynamic_variable,
                                model.gases, model.gravity, total_density, intermediate_momenta, intermediate_tracers, slow_source_terms)

            update_total_density!(total_density, model.grid, model.gases, intermediate_tracers)
            fill_halo_regions!(merge((Σρ=total_density,), intermediate_momenta, intermediate_tracers), model.architecture, model.clock, nothing)
        end

        fill_halo_regions!(momenta.ρw, model.architecture, model.clock, nothing)
        fill_halo_regions!(intermediate_momenta.ρw, model.architecture, model.clock, nothing)

        compute_fast_source_terms!(compute_rhs_args...)

        @debug "  Advancing variables..."
        LHS = rk3_iter == 3 ? Φ⁺ : intermediate_fields
        advance_variables!(LHS, model.grid, momenta, tracers, fast_source_terms; Δt=rk3_time_step(rk3_iter, Δt))
    end

    model.clock.iteration += 1
    model.clock.time += Δt

    return nothing
end
