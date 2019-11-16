using JULES.Operators

using Oceananigans: NoPenetrationBC, fill_halo_regions!, datatuple
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

@inline function RU(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FU)
    @inbounds begin
        return (- div_ρuũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p∂x(i, j, k, grid, pt, b, pₛ, C)
                + FU[i, j, k])
    end
end

@inline function RV(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FV)
    @inbounds begin
        return (- div_ρvũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * ∂p∂y(i, j, k, grid, pt, b, pₛ, C)
                + FV[i, j, k])
    end
end

@inline function RW(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C, FW)
    @inbounds begin
        return (- div_ρwũ(i, j, k, grid, ρᵈ, Ũ)
                - ρᵈ_over_ρᵐ(i, j, k, grid, ρᵈ, C) * (  ∂p∂z(i, j, k, grid, pt, b, pₛ, C)
                                                      + buoyancy_perturbation(i, j, k, grid, grav, ρᵈ, C))
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

function time_step!(model::CompressibleModel; Δt, Nt=1)
    arch = model.architecture
    grid = model.grid

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
        compute_slow_forcings!(F, grid, model.coriolis, Ũ, ρᵈ, C̃)
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

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F, grid, coriolis, Ũ, ρᵈ, C̃)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            F.U[i, j, k] = FU(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.V[i, j, k] = FV(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
            F.W[i, j, k] = FW(i, j, k, grid, coriolis, μ, ρᵈ, Ũ)
        end

        for C_name in propertynames(C̃)
            C   = getproperty(C̃, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                F_C[i, j, k] = FC(i, j, k, grid, κ, ρᵈ, C)
            end
        end
    end
end

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_right_hand_sides!(R, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            R.U[i, j, k] = RU(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.U)
            R.V[i, j, k] = RV(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.V)
            R.W[i, j, k] = RW(i, j, k, grid, ρᵈ, Ũ, pt, b, pₛ, C̃, F.W)

            R.ρ[i, j, k] = Rρ(i, j, k, grid, Ũ)
        end

        for C_name in propertynames(C̃)
            C   = getproperty(C̃, C_name)
            R_C = getproperty(R, C_name)
            F_C = getproperty(F, C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                R_C[i, j, k] = RC(i, j, k, grid, ρᵈ, Ũ, C, F_C)
            end
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
function advance_variables!(I, grid, Ũᵗ, C̃ᵗ, ρᵗ, R; Δt)
    @inbounds begin
        for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
            I.U[i, j, k] = Ũᵗ.U[i, j, k] + Δt * R.U[i, j, k]
            I.V[i, j, k] = Ũᵗ.V[i, j, k] + Δt * R.V[i, j, k]
            I.W[i, j, k] = Ũᵗ.W[i, j, k] + Δt * R.W[i, j, k]
            I.ρ[i, j, k] =   ρᵗ[i, j, k] + Δt * R.ρ[i, j, k]
        end

        for C_name in propertynames(C̃ᵗ)
            Cᵗ  = getproperty(C̃ᵗ, C_name)
            I_C = getproperty(I,  C_name)
            R_C = getproperty(R,  C_name)

            for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
                I_C[i, j, k] = Cᵗ[i, j, k] + Δt * R_C[i, j, k]
            end
        end
    end
end

