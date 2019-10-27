####
#### Utilities for acoustic time stepping
####

function acoustic_time_steps(rk3_iter, nₛ, Δt)
    rk3_iter == 1 && return 1,         Δt/3
    rk3_iter == 2 && return Int(nₛ/2), (Δt/2)/(nₛ/2)
    rk3_iter == 3 && return nₛ,        Δt/nₛ
end

acoustic_time_stepping!(args...) = nothing

####
#### Time-stepping algorithm
####

function time_step!(model; Δt, nₛ)
    arch = model.arch
    grid = model.grid

    Ũ = model.momenta
    ρ = model.densities
    F = model.slow_forcings
    R = model.fast_forcings

    C = model.tracers
    S = model.tracers.S

    # TODO: Fill halo regions for U, V, W, C
    compute_slow_forcings!(F, Ũ, ρ, C)

    # RK3 time-stepping
    for rk3_iter in 1:3
        # TODO: Fill halo regions for U, V, W, ρ, C
        compute_fast_forcings!(R, Ũ, ρ, F)

        n, Δτ = acoustic_time_steps(rk3_iter)
        acoustic_time_stepping!(Ũ, ρ, C, F, R; n=n, Δτ=Δτ)

        scalar_transport!(C, Ũ, args...)
    end

    return nothing
end

####
#### Calculation of "slow" forcings
####

@inline FU(i, j, k, ρ, Ũ) = @inbounds -(ρ.d[i, j, k] / ρ.m[i, j, k]) * x_f_cross_U(i, j, k, grid, rotation, Ũ) + μ∇²u(i, j, k, Ũ.U)
@inline FV(i, j, k, ρ, Ũ) = @inbounds -(ρ.d[i, j, k] / ρ.m[i, j, k]) * y_f_cross_U(i, j, k, grid, rotation, Ũ) + μ∇²v(i, j, k, Ũ.V)
@inline FW(i, j, k, ρ, Ũ) = @inbounds -(ρ.d[i, j, k] / ρ.m[i, j, k]) * z_f_cross_U(i, j, k, grid, rotation, Ũ) + μ∇²w(i, j, k, Ũ.W)

@inline FC(i, j, k, C) = κ∇²(i, j, k, C)

"""
Slow forcings include viscous dissipation, diffusion, and Coriolis terms.
"""
function compute_slow_forcings!(F, Ũ, ρ, C)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds F.U[i, j, k] = FU(i, j, k, ρ, Ũ)
        @inbounds F.V[i, j, k] = FV(i, j, k, ρ, Ũ)
        @inbounds F.W[i, j, k] = FW(i, j, k, ρ, Ũ)

        @inbounds F.S[i, j, k]  = FC(i, j, k, C.S)
        @inbounds F.Qv[i, j, k] = FC(i, j, k, C.Qv)
        @inbounds F.Ql[i, j, k] = FC(i, j, k, C.Ql)
        @inbounds F.Qi[i, j, k] = FC(i, j, k, C.Qi)
    end
end

####
#### Calculation of "fast" forcings
####

@inline buoyancy_perturbation(i, j, k, buoyancy::DryIdealGas, args...) = nothing

@inline RU(i, j, k, Ũ, ρ, S, FU) = @inbounds -∇ρũu(i, j, k, Ũ) - (ρ.d[i, j, k]/ρ.m[i, j, k]) * ∂x_p′(i, j, k, S, ρ) + FU[i, j, k]
@inline RV(i, j, k, Ũ, ρ, S, FU) = @inbounds -∇ρũv(i, j, k, Ũ) - (ρ.d[i, j, k]/ρ.m[i, j, k]) * ∂y_p′(i, j, k, S, ρ) + FV[i, j, k]
@inline RW(i, j, k, Ũ, ρ, S, FU) = @inbounds -∇ρũw(i, j, k, Ũ) - (ρ.d[i, j, k]/ρ.m[i, j, k]) * (∂z_p′(i, j, k, S, ρ) + buoyancy_perturbation(i, j, k, B, args...)) + FW[i, j, k]

@inline Rρd(i, j, k, Ũ) = -div(i, j, k, Ũ)

@inline RC(i, j, k, Ũ, C) = -div_flux(i, j, k, Ũ, C)

"""
Fast forcings include advection, pressure gradient, and buoyancy terms.
"""
function compute_fast_forcings!(R, Ũ, ρ, F)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        @inbounds R.U[i, j, k] = RU(i, j, k, ρ, Ũ)
        @inbounds R.V[i, j, k] = RV(i, j, k, ρ, Ũ)
        @inbounds R.W[i, j, k] = RW(i, j, k, ρ, Ũ)
        @inbounds R.ρd[i, j, k] = Rρd(i, j, k, Ũ)

        @inbounds R.S[i, j, k]  = RC(i, j, k, Ũ, C.S)
        @inbounds R.Qv[i, j, k] = RC(i, j, k, Ũ, C.Qv)
        @inbounds R.Ql[i, j, k] = RC(i, j, k, Ũ, C.Ql)
        @inbounds R.Qi[i, j, k] = RC(i, j, k, Ũ, C.Qi)
    end
end

