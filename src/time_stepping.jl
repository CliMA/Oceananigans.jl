using JULES.Operators

using Oceananigans: datatuple
using Oceananigans.BoundaryConditions
import Oceananigans: time_step!

####
#### Utilities for time stepping
####

function rk3_time_step(rk3_iter, Δt)
    rk3_iter == 1 && return Δt/3
    rk3_iter == 2 && return Δt/2
    rk3_iter == 3 && return Δt
end

####
#### Time-stepping algorithm
####

function time_step!(model::CompressibleModel; Δt, Nt=1)

    arch = model.architecture
    grid = model.grid
    time = model.clock.time
    coriolis = model.coriolis
    closure = model.closure
    tvar = model.thermodynamic_variable
    microphysics = model.microphysics
    forcing = model.forcing

    Ũ  = model.momenta
    ρ  = model.total_density
    ρ̃  = model.gases
    C̃  = model.tracers
    K̃  = model.diffusivities
    F  = model.slow_forcings
    R  = model.right_hand_sides
    IV = model.intermediate_vars

    g  = model.gravity

    # On third RK3 step, we update Φ⁺ instead of model.intermediate_vars
    Φ⁺ = (Ũ..., tracers = C̃)

    # On the first and second RK3 steps we want to update intermediate Ũ and C̃.
    Ũ_names = propertynames(Ũ)
    IV_Ũ_vals = [getproperty(IV, U) for U in Ũ_names]
    IV_Ũ = NamedTuple{Ũ_names}(IV_Ũ_vals)

    C̃_names = propertynames(C̃)
    IV_C̃_vals = [getproperty(IV.tracers, C) for C in C̃_names]
    IV_C̃ = NamedTuple{C̃_names}(IV_C̃_vals)

    for _ in 1:Nt
        @debug "Computing slow forcings..."
        update_total_density!(ρ.data, grid, ρ̃, C̃)
        fill_halo_regions!(merge((Σρ=ρ,), Ũ, C̃), arch)
        compute_slow_forcings!(F, grid, tvar, g, coriolis, closure, Ũ, ρ, ρ̃, C̃, K̃, forcing, time)
        fill_halo_regions!(F.ρw, arch)

        # RK3 time-stepping
        for rk3_iter in 1:3
            @debug "RK3 step #$rk3_iter..."

            @debug "  Computing right hand sides..."
            if rk3_iter == 1
                compute_rhs_args = (R, grid, tvar, g, ρ, ρ̃, Ũ, C̃, F)
                update_total_density!(ρ.data, grid, ρ̃, C̃)
                fill_halo_regions!(merge((Σρ=ρ,), Ũ, C̃), arch)
            else
                compute_rhs_args = (R, grid, tvar, g, ρ, ρ̃, IV_Ũ, IV_C̃, F)
                update_total_density!(ρ.data, grid, ρ̃, IV_C̃)
                fill_halo_regions!(merge((Σρ=ρ,), IV_Ũ, IV_C̃), arch)
            end

            compute_right_hand_sides!(compute_rhs_args...)

            # n, Δτ = acoustic_time_steps(rk3_iter)
            # acoustic_time_stepping!(Ũ, ρ, C, F, R; n=n, Δτ=Δτ)

            @debug "  Advancing variables..."
            LHS = rk3_iter == 3 ? Φ⁺ : IV
            advance_variables!(LHS, grid, Ũ, C̃, R; Δt=rk3_time_step(rk3_iter, Δt))
        end

        model.clock.iteration += 1
        model.clock.time += Δt
    end

    return nothing
end
