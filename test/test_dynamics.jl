xnodes(ϕ) = repeat(reshape(ϕ.grid.xC, ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ) = repeat(reshape(ϕ.grid.yC, 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ) = repeat(reshape(ϕ.grid.zC, 1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

xnodes(ϕ::FaceFieldX) = repeat(reshape(ϕ.grid.xF[1:end-1], ϕ.grid.Nx, 1, 1), 1, ϕ.grid.Ny, ϕ.grid.Nz)
ynodes(ϕ::FaceFieldY) = repeat(reshape(ϕ.grid.yF[1:end-1], 1, ϕ.grid.Ny, 1), ϕ.grid.Nx, 1, ϕ.grid.Nz)
znodes(ϕ::FaceFieldZ) = repeat(reshape(ϕ.grid.zF[2:end],   1, 1, ϕ.grid.Nz), ϕ.grid.Nx, ϕ.grid.Ny, 1)

zerofunk(args...) = 0

function set_ic!(model; u=zerofunk, v=zerofunk, w=zerofunk, T=zerofunk, S=zerofunk)
    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    data(model.velocities.u) .= u.(xnodes(model.velocities.u), ynodes(model.velocities.u), znodes(model.velocities.u))
    data(model.velocities.v) .= v.(xnodes(model.velocities.v), ynodes(model.velocities.v), znodes(model.velocities.v))
    data(model.velocities.w) .= w.(xnodes(model.velocities.w), ynodes(model.velocities.w), znodes(model.velocities.w))
    data(model.tracers.T)    .= T.(xnodes(model.tracers.T),    ynodes(model.tracers.T),    znodes(model.tracers.T))
    data(model.tracers.S)    .= S.(xnodes(model.tracers.S),    ynodes(model.tracers.S),    znodes(model.tracers.S))
    return nothing
end

function u_relative_error(model, u)
    u_num = model.velocities.u

    u_ans = FaceFieldX(u.(
        xnodes(u_num), ynodes(u_num), znodes(u_num), model.clock.time),
        model.grid)

    return mean(
        (data(u_num) .- u_ans.data).^2 ) / mean(u_ans.data.^2)
end

function w_relative_error(model, w)
    w_num = model.velocities.w

    w_ans = FaceFieldZ(
        w.(xnodes(w_num), ynodes(w_num), znodes(w_num), model.clock.time),
        model.grid)

    return mean(
        (data(w_num) .- w_ans.data).^2 ) / mean(w_ans.data.^2)
end

function T_relative_error(model, T)
    T_num = model.tracers.T

    T_ans = FaceFieldZ(
        T.(xnodes(T_num), ynodes(T_num), znodes(T_num), model.clock.time),
        model.grid)

    return mean(
        (data(T_num) .- T_ans.data).^2 ) / mean(T_ans.data.^2)
end

function test_diffusion_simple(fld)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 1
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if fld ∈ (:u, :v, :w)
        field = getfield(model.velocities, fld)
    else
        field = getfield(model.tracers, fld)
    end

    value = π
    field.data .= value

    Δt = 0.01 # time-step much less than diffusion time-scale
    Nt = 10
    time_step!(model, Nt, Δt)

    !any(@. !isapprox(value, field.data))
end


function test_diffusion_budget(field_name)
    Nx, Ny, Nz = 1, 1, 16
    Lx, Ly, Lz = 1, 1, 1
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if field_name ∈ (:u, :v, :w)
        field = getfield(model.velocities, field_name)
    else
        field = getfield(model.tracers, field_name)
    end

    half_Nz = floor(Int, Nz/2)
    data(field)[:, :,   1:half_Nz] .= -1
    data(field)[:, :, half_Nz:end] .=  1

    mean_init = mean(field.data)
    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 0.0001 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)
    isapprox(mean_init, mean(field.data))
end

function test_diffusion_cosine(fld)
    Nx, Ny, Nz = 1, 1, 128
    Lx, Ly, Lz = 1, 1, π/2
    κ = 1
    eos = LinearEquationOfState(βS=0, βT=0)

    model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=κ, κ=κ, eos=eos)

    if fld == :w
        throw("There are no boundary condition tests for w yet.")
    elseif fld ∈ (:u, :v)
        field = getfield(model.velocities, fld)
    else
        field = getfield(model.tracers, fld)
    end

    zC = model.grid.zC
    m = 2
    data(field)[1, 1, :] .= cos.(m*zC)

    diffusing_cosine(κ, m, z, t) = exp(-κ*m^2*t) * cos(m*z)

    τκ = Lz^2 / κ # diffusion time-scale
    Δt = 1e-6 * τκ # time-step much less than diffusion time-scale
    Nt = 100

    time_step!(model, Nt, Δt)

    field_numerical = dropdims(data(field), dims=(1, 2))

    !any(@. !isapprox(field_numerical, diffusing_cosine(κ, m, zC, model.clock.time), atol=1e-6, rtol=1e-6))
end


function internal_wave_test(; N=128, Nt=10)

    # Internal wave parameters
     ν = κ = 1e-9
     L = 2π
    z₀ = -L/3
     δ = L/20
    a₀ = 1e-3
     m = 16
     k = 1
     f = 0.2
     ℕ = 1.0
     σ = sqrt( (ℕ^2*k^2 + f^2*m^2) / (k^2 + m^2) )

    # Numerical parameters
     N = 128
    Δt = 0.01 * 1/σ

    cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
     U = a₀ * k * σ   / (σ^2 - f^2)
     V = a₀ * k * f   / (σ^2 - f^2)
     W = a₀ * m * σ   / (σ^2 - ℕ^2)
     Θ = a₀ * m * ℕ^2 / (σ^2 - ℕ^2)

    a(x, y, z, t) = exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )

    u(x, y, z, t) =           a(x, y, z, t) * U * cos(k*x + m*z - σ*t)
    v(x, y, z, t) =           a(x, y, z, t) * V * sin(k*x + m*z - σ*t)
    w(x, y, z, t) =           a(x, y, z, t) * W * cos(k*x + m*z - σ*t)
    T(x, y, z, t) = ℕ^2 * z + a(x, y, z, t) * Θ * sin(k*x + m*z - σ*t)

    u₀(x, y, z) = u(x, y, z, 0)
    v₀(x, y, z) = v(x, y, z, 0)
    w₀(x, y, z) = w(x, y, z, 0)
    T₀(x, y, z) = T(x, y, z, 0)

    # Create a model where temperature = buoyancy.
    model = Model(N=(N, 1, N), L=(L, L, L), ν=ν, κ=κ,
                    eos=LinearEquationOfState(βT=1.),
                    constants=PlanetaryConstants(f=f, g=1.))

    set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀)

    time_step!(model, Nt, Δt)

    # Tolerance was found by trial and error...
    u_relative_error(model, u) < 1e-4
end

function passive_tracer_advection_test(; N=128, κ=1e-12, Nt=100)
    L, U, V = 1.0, 0.5, 0.8
    δ, x₀, y₀ = L/15, L/2, L/2

    Δt = 0.05 * L/N / sqrt(U^2 + V^2)

    T(x, y, z, t) = exp( -((x - U*t - x₀)^2 + (y - V*t - y₀)^2) / (2*δ^2) )
    u₀(x, y, z) = U
    v₀(x, y, z) = V
    T₀(x, y, z) = T(x, y, z, 0)

    model = Model(N=(N, N, 2), L=(L, L, L), ν=κ, κ=κ)

    set_ic!(model, u=u₀, v=v₀, T=T₀)
    time_step!(model, Nt, Δt)

    # Error tolerance is a bit arbitrary
    return T_relative_error(model, T) < 1e-4
end
