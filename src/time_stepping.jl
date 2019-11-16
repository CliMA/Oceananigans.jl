using JULES.Operators

using Oceananigans: NoPenetrationBC, fill_halo_regions!, datatuple
import Oceananigans: time_step!

####
#### Dirty hacks!
####

const hpbcs = HorizontallyPeriodicBCs()
const hpbcs_np = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())

####
#### Utilities for time stepping
####

function rk3_time_step(rk3_iter, Δt)
    rk3_iter == 1 && return Δt/3
    rk3_iter == 2 && return Δt/2
    rk3_iter == 3 && return Δt
end

function acoustic_time_steps(rk3_iter, nₛ, Δt)
    rk3_iter == 1 && return 1,         Δt/3
    rk3_iter == 2 && return Int(nₛ/2), Δt/nₛ
    rk3_iter == 3 && return nₛ,        Δt/nₛ
end

####
#### Time-stepping algorithm
####

function time_step!(model::CompressibleModel; Δt, Nt=1)
    arch = model.architecture
    grid = model.grid
    time = model.clock.time
    forcing = model.forcing
    params = model.parameters

    Ũ = model.momenta
    C̃ = model.tracers
    F = model.slow_forcings
    R = model.right_hand_sides
    IV = model.intermediate_vars

    ρᵈ = model.density
    Θᵐ = model.tracers.Θᵐ

    pₛ = model.reference_pressure

    # On third RK3 step, we update Φ⁺ instead of model.intermediate_vars
    Φ⁺ = merge(Ũ, C̃, (ρ=ρᵈ,))

    # On the first and second RK3 steps we can to update intermediate Ũ and C̃.
    Ũ_names = propertynames(Ũ)
    IV_Ũ_vals = [getproperty(IV, U) for U in Ũ_names]
    IV_Ũ = NamedTuple{Ũ_names}(IV_Ũ_vals)

    C̃_names = propertynames(C̃)
    IV_C̃_vals = [getproperty(IV, C) for C in C̃_names]
    IV_C̃ = NamedTuple{C̃_names}(IV_C̃_vals)

    for _ in 1:Nt
        @debug "Computing slow forcings..."
        fill_halo_regions!(ρᵈ.data, hpbcs, arch, grid)
        fill_halo_regions!(datatuple(merge(Ũ, C̃)), hpbcs, arch, grid)
        fill_halo_regions!(Ũ.W.data, hpbcs_np, arch, grid)
        compute_slow_forcings!(F, grid, model.coriolis, Ũ, ρᵈ, C̃, forcing, time, params)
        fill_halo_regions!(F.W.data, hpbcs_np, arch, grid)

        # RK3 time-stepping
        for rk3_iter in 1:3
            @debug "RK3 step #$rk3_iter..."

            @debug "  Computing right hand sides..."
            if rk3_iter == 1
                compute_rhs_args = (R, grid, ρᵈ, Ũ, model.prognostic_temperature, model.buoyancy, pₛ, C̃, F)

                fill_halo_regions!(ρᵈ.data, hpbcs, arch, grid)
                fill_halo_regions!(datatuple(merge(Ũ, C̃)), hpbcs, arch, grid)
                fill_halo_regions!(Ũ.W.data, hpbcs_np, arch, grid)
            else
                compute_rhs_args = (R, grid, IV.ρ, IV_Ũ, model.prognostic_temperature, model.buoyancy, pₛ, IV_C̃, F)

                fill_halo_regions!(IV.ρ.data, hpbcs, arch, grid)
                fill_halo_regions!(datatuple(merge(IV_Ũ, IV_C̃)), hpbcs, arch, grid)
                fill_halo_regions!(IV_Ũ.W.data, hpbcs_np, arch, grid)
            end

            compute_right_hand_sides!(compute_rhs_args...)

            fill_halo_regions!(R.W.data, hpbcs_np, arch, grid)

            # n, Δτ = acoustic_time_steps(rk3_iter)
            # acoustic_time_stepping!(Ũ, ρ, C, F, R; n=n, Δτ=Δτ)

            @debug "  Advancing variables..."
            LHS = rk3_iter == 3 ? Φ⁺ : IV
            advance_variables!(LHS, grid, Ũ, C̃, ρᵈ, R; Δt=rk3_time_step(rk3_iter, Δt))
        end

        model.clock.iteration += 1
        model.clock.time += Δt
    end

    return nothing
end
