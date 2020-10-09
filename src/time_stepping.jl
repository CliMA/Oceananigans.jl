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

# Adding kwargs... so this time_step! can work with Oceananigans.Simulation
function time_step!(model::CompressibleModel, Δt)
    ρ  = model.total_density
    ρũ = model.momenta
    ρc̃ = model.tracers
    K̃  = model.diffusivities
    F̃  = model.slow_forcings
    R̃  = model.right_hand_sides
    IV = model.intermediate_variables

    # On third RK3 step, we update Φ⁺ instead of model.intermediate_variables
    Φ⁺ = (ρũ..., tracers = ρc̃)

    # On the first and second RK3 steps we want to update intermediate ρũ and ρc̃.
    ρũ_names = propertynames(ρũ)
    ρc̃_names = propertynames(ρc̃)

    IV_ρũ_fields = [getproperty(IV, ρu)         for ρu in ρũ_names]
    IV_ρc̃_fields = [getproperty(IV.tracers, ρc) for ρc in ρc̃_names]

    IV_ρũ = NamedTuple{ρũ_names}(IV_ρũ_fields)
    IV_ρc̃ = NamedTuple{ρc̃_names}(IV_ρc̃_fields)

    @debug "Computing slow forcings..."
    update_total_density!(ρ, model.grid, model.gases, ρc̃)
    fill_halo_regions!(merge((Σρ=ρ,), ρũ, ρc̃), model.architecture, model.clock, nothing)

    fill_halo_regions!(ρũ.ρw, model.architecture, model.clock, nothing)
    fill_halo_regions!(IV_ρũ.ρw, model.architecture, model.clock, nothing)

    compute_slow_forcings!(
        F̃, model.grid, model.thermodynamic_variable, model.gases, model.gravity,
        model.coriolis, model.closure, ρ, ρũ, ρc̃, K̃, model.forcing, model.clock)

    fill_halo_regions!(F̃.ρw, model.architecture, model.clock, nothing)

    for rk3_iter in 1:3
        @debug "RK3 step #$rk3_iter..."
        @debug "  Computing right hand sides..."

        if rk3_iter == 1
            compute_rhs_args = (R̃, model.grid, model.thermodynamic_variable,
                                model.gases, model.gravity, ρ, ρũ, ρc̃, F̃)

            update_total_density!(ρ, model.grid, model.gases, ρc̃)
            fill_halo_regions!(merge((Σρ=ρ,), ρũ, ρc̃), model.architecture, model.clock, nothing)
        else
            compute_rhs_args = (R̃, model.grid, model.thermodynamic_variable,
                                model.gases, model.gravity, ρ, IV_ρũ, IV_ρc̃, F̃)

            update_total_density!(ρ, model.grid, model.gases, IV_ρc̃)
            fill_halo_regions!(merge((Σρ=ρ,), IV_ρũ, IV_ρc̃), model.architecture, model.clock, nothing)
        end

        fill_halo_regions!(ρũ.ρw, model.architecture, model.clock, nothing)
        fill_halo_regions!(IV_ρũ.ρw, model.architecture, model.clock, nothing)

        compute_right_hand_sides!(compute_rhs_args...)

        @debug "  Advancing variables..."
        LHS = rk3_iter == 3 ? Φ⁺ : IV
        advance_variables!(LHS, model.grid, ρũ, ρc̃, R̃; Δt=rk3_time_step(rk3_iter, Δt))
    end

    model.clock.iteration += 1
    model.clock.time += Δt

    return nothing
end
