using JULES.Operators

using Oceananigans: fill_halo_regions!, datatuple
import Oceananigans: time_step!

####
#### Dirty hacks!
####

const grav = 9.80665
const μ = 1e-2
const κ = 1e-2

const hpbcs = HorizontallyPeriodicBCs()
const hpbcs_np = HorizontallyPeriodicBCs(top=NoPenetrationBC(), bottom=NoPenetrationBC())

####
#### Element-wise forcing and right-hand-side calculations
####

@inline FU(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -x_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇u(i, j, k, grid, μ, ρᵈ, Ũ.U)
@inline FV(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -y_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇v(i, j, k, grid, μ, ρᵈ, Ũ.V)
@inline FW(i, j, k, grid, coriolis, μ, ρᵈ, Ũ) = -z_f_cross_U(i, j, k, grid, coriolis, Ũ) + div_μ∇w(i, j, k, grid, μ, ρᵈ, Ũ.W)

@inline FC(i, j, k, grid, κ, ρᵈ, C) = div_κ∇c(i, j, k, grid, κ, ρᵈ, C)

@inline function RU(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, FU, base_state)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p′∂x(i, j, k, grid, pt, b, p₀, C, base_state)
                + FU[i, j, k])
    end
end

@inline function RV(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, FV, base_state)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p′∂y(i, j, k, grid, pt, b, p₀, C, base_state)
                + FV[i, j, k])
    end
end

@inline function RW(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, FW, base_state)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * (  ∂p′∂z(i, j, k, grid, pt, b, p₀, C, base_state)
                                                      + buoyancy_perturbation(i, j, k, grid, grav, ρᵈ, C, base_state))
                + FW[i, j, k])
    end
end

@inline Rρ(i, j, k, grid, Ũ) = -divᶜᶜᶜ(i, j, k, grid, Ũ.U, Ũ.V, Ũ.W)
@inline RC(i, j, k, grid, ρᵈ, Ũ, C, FC) = @inbounds -div_flux(i, j, k, grid, ρᵈ, Ũ.U, Ũ.V, Ũ.W, C) + FC[i, j, k]

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

acoustic_time_stepping!(args...) = nothing

####
#### Time-stepping algorithm
####

function time_step!(model::CompressibleModel; Δt, nₛ)
    arch = model.architecture
    grid = model.grid

    Ũ = model.momenta
    C = model.tracers
    F = model.slow_forcings
    R = model.right_hand_sides
    IV = model.intermediate_vars

    ρᵈ = model.density
    Θᵐ = model.tracers.Θᵐ

    p₀ = model.surface_pressure
    BS = model.base_state

    # On third RK3 step, we update Φ⁺ instead of model.intermediate_vars
    Φ⁺ = (U=Ũ.U, V=Ũ.V, W=Ũ.W, ρ=ρᵈ, Θᵐ=Θᵐ, Qv=C.Qv, Ql=C.Ql, Qi=C.Qi)

    @debug "Computing slow forcings..."
    fill_halo_regions!(ρᵈ.data, hpbcs, arch, grid)
    fill_halo_regions!(datatuple(merge(Ũ, C)), hpbcs, arch, grid)
    fill_halo_regions!(Ũ.W, hpbcs_np, arch, grid)
    compute_slow_forcings!(F, grid, model.coriolis, Ũ, ρᵈ, C)

    # RK3 time-stepping
    for rk3_iter in 1:3
        @debug "RK3 step #$rk3_iter..."

        @debug "  Computing right hand sides..."
        if rk3_iter == 1
            compute_rhs_args = (R, grid, ρᵈ, Ũ, model.prognostic_temperature, model.buoyancy, p₀, C, F, BS)
            fill_halo_regions!(ρᵈ.data, hpbcs, arch, grid)
            fill_halo_regions!(datatuple(merge(Ũ, C)), hpbcs, arch, grid)
            fill_halo_regions!(Ũ.W, hpbcs_np, arch, grid)
        else
            IV_Ũ = (U=IV.U, V=IV.V, W=IV.W)
            IV_C = (Θᵐ=IV.Θᵐ, Qv=IV.Qv, Ql=IV.Ql, Qi=IV.Qi)
            compute_rhs_args = (R, grid, IV.ρ, IV_Ũ, model.prognostic_temperature, model.buoyancy, p₀, IV_C, F, BS)
            fill_halo_regions!(IV.ρ.data, hpbcs, arch, grid)
            fill_halo_regions!(datatuple(merge(IV_Ũ, IV_C)), hpbcs, arch, grid)
            fill_halo_regions!(IV_Ũ.W, hpbcs_np, arch, grid)
        end

        compute_right_hand_sides!(compute_rhs_args...)

        # n, Δτ = acoustic_time_steps(rk3_iter)
        # acoustic_time_stepping!(Ũ, ρ, C, F, R; n=n, Δτ=Δτ)

        @debug "  Advancing variables..."
        LHS = rk3_iter == 3 ? Φ⁺ : IV
        advance_variables!(LHS, grid, Ũ, C, ρᵈ, R; Δt=rk3_time_step(rk3_iter, Δt))
    end

    model.clock.iteration += 1
    model.clock.time += Δt

    return nothing
end

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F, grid, coriolis, Ũ, ρᵈ, C)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            F.U[i, j, k] = FU(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.V[i, j, k] = FV(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.W[i, j, k] = FW(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)

            F.Θᵐ[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C.Θᵐ)
            F.Qv[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C.Qv)
            F.Ql[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C.Ql)
            F.Qi[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C.Qi)
        end
    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_right_hand_sides!(R, grid, ρᵈ, Ũ, pt, b, p₀, C, F, base_state)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R.U[i, j, k] = RU(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, F.U, base_state)
            R.V[i, j, k] = RV(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, F.V, base_state)
            R.W[i, j, k] = RW(i, j, k, grid, ρᵈ, Ũ, pt, b, p₀, C, F.W, base_state)

            R.ρ[i, j, k] = Rρ(i, j, k, grid, Ũ)

            R.Θᵐ[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C.Θᵐ, F.Θᵐ)
            R.Qv[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C.Qv, F.Qv)
            R.Ql[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C.Ql, F.Ql)
            R.Qi[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C.Qi, F.Qi)
        end
    end
end

####
#### Advancing variables
####

"""
Updates variables according to the RK3 time step:
    1. Φ*      = Φᵗ + Δt/3 * R(Φᵗ)
    2. Φ**     = Φᵗ + Δt/2 * R(Φ*)
    3. Φ(t+Δt) = Φᵗ + Δt   * R(Φ**)
"""
function advance_variables!(I, grid, Ũᵗ, Cᵗ, ρᵗ, R; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            I.U[i, j, k] = Ũᵗ.U[i, j, k] + Δt * R.U[i, j, k]
            I.V[i, j, k] = Ũᵗ.V[i, j, k] + Δt * R.V[i, j, k]
            I.W[i, j, k] = Ũᵗ.W[i, j, k] + Δt * R.W[i, j, k]
            I.ρ[i, j, k] =   ρᵗ[i, j, k] + Δt * R.ρ[i, j, k]

            I.Θᵐ[i, j, k] = Cᵗ.Θᵐ[i, j, k] + Δt * R.Θᵐ[i, j, k]
            I.Qv[i, j, k] = Cᵗ.Qv[i, j, k] + Δt * R.Qv[i, j, k]
            I.Ql[i, j, k] = Cᵗ.Ql[i, j, k] + Δt * R.Ql[i, j, k]
            I.Qi[i, j, k] = Cᵗ.Qi[i, j, k] + Δt * R.Qi[i, j, k]
        end
    end
end

