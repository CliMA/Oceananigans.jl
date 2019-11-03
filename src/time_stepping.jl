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

function time_step!(model; Δt, nₛ)
    arch = model.arch
    grid = model.grid

    Ũ = model.momenta
    ρ = model.densities
    F = model.slow_forcings
    R = model.fast_forcings

    C = model.tracers
    S = model.tracers.S

    # Φ⁺ = (U=Ũ.U, V=Ũ.V, W=Ũ.W, ...)

    # TODO: Fill halo regions for U, V, W, C
    compute_slow_forcings!(F, Ũ, ρ, C)

    # RK3 time-stepping
    for rk3_iter in 1:3
        # TODO: Fill halo regions for U, V, W, ρ, C
        compute_fast_forcings!(R, Ũ, ρ, F)

        # n, Δτ = acoustic_time_steps(rk3_iter)
        # acoustic_time_stepping!(Ũ, ρ, C, F, R; n=n, Δτ=Δτ)

        I = rk3_iter == 3 ? Φ⁺ : model.intermediate_vars
        advance_variables!(I, Ũ, C, ρ, R; Δt=rk3_time_step(rk3_iter))
    end

    return nothing
end

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

####
#### Advancing variables
####

"""
Updates variables according to the RK3 time step:
    1. Φ*      = Φᵗ + Δt/3 * R(Φᵗ)
    2. Φ**     = Φᵗ + Δt/2 * R(Φ*)
    3. Φ(t+Δt) = Φᵗ + Δt   * R(Φ**)
"""
function advance_variables!(IV, Ũᵗ, Cᵗ, ρᵗ, R; Δt)
    for k in 1:grid.Nz, j in 1:grid.Ny, i in 1:grid.Nx
        I.U[i, j, k] = Ũᵗ.U[i, j, k] + Δt * R.U[i, j, k]
        I.V[i, j, k] = Ũᵗ.V[i, j, k] + Δt * R.V[i, j, k]
        I.W[i, j, k] = Ũᵗ.W[i, j, k] + Δt * R.W[i, j, k]
        I.ρ[i, j, k] = ρᵗ.d[i, j, k] + Δt * R.ρd[i, j, k]
        I.S[i, j, k] = Cᵗ.S[i, j, k] + Δt * R.S[i, j, k]

        I.Qv[i, j, k] = Cᵗ.Qv[i, j, k] + Δt * R.Qv[i, j, k]
        I.Ql[i, j, k] = Cᵗ.Ql[i, j, k] + Δt * R.Ql[i, j, k]
        I.Qi[i, j, k] = Cᵗ.Qi[i, j, k] + Δt * R.Qi[i, j, k]
    end
end

