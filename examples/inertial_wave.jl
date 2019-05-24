using Oceananigans, Printf, PyPlot

include("utils.jl")

# Numerical parameters
Nx, Ny, Nz = 256, 1, 256
 L = 2π
 f = 1.0
Δt = 0.01
ν = κ = 1e-9

# Wave parameters
z₀ = -L/2
a₀ = 1e-9
 m = 12
 k = 8
@show σ = f*m/sqrt(k^2 + m^2)
 δ = L/15

# Analytical solution for an inviscid inertial wave
cᵍ = m * σ / (k^2 + m^2) * (f^2/σ^2 - 1)
 𝒰 = k * σ / (σ^2 - f^2)
 𝒱 = k * f / (σ^2 - f^2)
 𝒲 = m / σ

a(x, y, z, t) = a₀ * exp( -(z - cᵍ*t - z₀)^2 / (2*δ)^2 )
u(x, y, z, t) = a(x, y, z, t) * 𝒰 * cos(k*x + m*z - σ*t)
v(x, y, z, t) = a(x, y, z, t) * 𝒱 * sin(k*x + m*z - σ*t)
w(x, y, z, t) = a(x, y, z, t) * 𝒲 * cos(k*x + m*z - σ*t)

u₀(x, y, z) = u(x, y, z, 0)
v₀(x, y, z) = v(x, y, z, 0)
w₀(x, y, z) = w(x, y, z, 0)

function makeplot(axs, model)

    w_ans = FaceFieldZ(w.(
        xnodes(model.velocities.w),
        ynodes(model.velocities.w),
        znodes(model.velocities.w),
        model.clock.time), model.grid)

    u_ans = FaceFieldX(u.(
        xnodes(model.velocities.u),
        ynodes(model.velocities.u),
        znodes(model.velocities.u),
        model.clock.time), model.grid)

    sca(axs[1])
    PyPlot.plot(w_ans.data[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(model.velocities.w.data[1, 1, :], "--", linewidth=1)

    sca(axs[2])
    PyPlot.plot(model.velocities.w.data[1, 1, :] .- w_ans.data[1, 1, :])

    sca(axs[3])
    PyPlot.plot(u_ans.data[1, 1, :], "-", linewidth=2, alpha=0.4)
    PyPlot.plot(model.velocities.u.data[1, 1, :], "--", linewidth=1)

    sca(axs[4])
    PyPlot.plot(model.velocities.u.data[1, 1, :] .- u_ans.data[1, 1, :])

    return nothing
end

function w_relative_error(model)

    w_ans = FaceFieldZ(w.(
        xnodes(model.velocities.w),
        ynodes(model.velocities.w),
        znodes(model.velocities.w),
        model.clock.time), model.grid)

    return mean(
        (model.velocities.w.data[1, 1, :] .- w_ans.data[1, 1, :]).^2
    ) / mean(w_ans.data[1, 1, :].^2)
end

function u_relative_error(model)

    u_ans = FaceFieldX(u.(
        xnodes(model.velocities.u),
        ynodes(model.velocities.u),
        znodes(model.velocities.u),
        model.clock.time), model.grid)

    return mean(
        (model.velocities.u.data[1, 1, :] .- u_ans.data[1, 1, :]).^2
    ) / mean(u_ans.data[1, 1, :].^2)
end

# Create the model.
model = Model(N=(Nx, Ny, Nz), L=(L, L, L), ν=ν, κ=κ, constants=PlanetaryConstants(f=f))

set_ic!(model, u=u₀, v=v₀, w=w₀)

fig, axs = subplots(nrows=4, figsize=(6, 8))

time_step!(model, 5500, Δt)
makeplot(axs, model)
@show w_relative_error(model)
@show u_relative_error(model)

#time_step!(model, 1500, Δt)
#makeplot(axs, model)
#@show w_relative_error(model)
#@show u_relative_error(model)
gcf()
