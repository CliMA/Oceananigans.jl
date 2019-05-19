xnodes(ϕ) = repeat(reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ) = repeat(reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ) = repeat(reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

xnodes(ϕ::FaceFieldX) = repeat(reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ::FaceFieldY) = repeat(reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ::FaceFieldZ) = repeat(reshape(ϕ.grid.zF[2:end],   1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

zerofunk(args...) = 0

function set_ic!(model; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk)
    model.velocities.u.data .= u.(xnodes(model.velocities.u), ynodes(model.velocities.u), znodes(model.velocities.u))
    model.velocities.v.data .= v.(xnodes(model.velocities.v), ynodes(model.velocities.v), znodes(model.velocities.v))
    model.velocities.w.data .= w.(xnodes(model.velocities.w), ynodes(model.velocities.w), znodes(model.velocities.w))
    model.tracers.T.data    .= T.(xnodes(model.tracers.T),    ynodes(model.tracers.T),    znodes(model.tracers.T))
    model.tracers.S.data    .= S.(xnodes(model.tracers.S),    ynodes(model.tracers.S),    znodes(model.tracers.S))
    return nothing
end

function inertial_wave_test(; N=256, Δt=0.01, κ=1e-6, m=12, k=8, Nt=100)

    # Numerical parameters
     L = 2π
     f = 1.0

    # Wave parameters
    z₀ = -L/2
    a₀ = 1e-9
     σ = f*m/sqrt(k^2 + m^2)
     δ = L/15

    # Analytical solution for an inviscid inertial wave
    cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
     U = k * σ / (σ^2 - f^2)
     V = k * f / (σ^2 - f^2)
     W = m / σ

    a(x, y, z, t) = a₀ * exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )
    u(x, y, z, t) = a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) = a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) = a(x, y, z, t) * W * cos(k*x + m*z - σ*t)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)

    function w_relative_error(model)
        w_num = model.velocities.w

        w_ans = FaceFieldZ(
            w.(xnodes(w_num), ynodes(w_num), znodes(w_num), model.clock.time),
            model.grid)

        return mean(
            (w_num.data[1, 1, :] .- w_ans.data[1, 1, :]).^2 ) / mean(w_ans.data[1, 1, :].^2)
    end

    function u_relative_error(model)
        u_num = model.velocities.u

        u_ans = FaceFieldX(u.(
            xnodes(u_num), ynodes(u_num), znodes(u_num), model.clock.time),
            model.grid)

        return mean(
            (u_num.data[1, 1, :] .- u_ans.data[1, 1, :]).^2 ) / mean(u_ans.data[1, 1, :].^2)
    end

    # Create the model.
    model = Model(N=(N, 1, N), L=(L, L, L), ν=κ, κ=κ, constants=PlanetaryConstants(f=f))

    set_ic!(model, u=u₀, v=v₀, w=w₀)
    time_step!(model, Nt, Δt)

    # Error tolerance is a bit arbitrary
    u_relative_error(model) < 1e-2
end
