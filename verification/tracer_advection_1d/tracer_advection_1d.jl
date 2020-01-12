using Printf
using OffsetArrays
using Plots

abstract type AbstractAdvectionScheme end

# Second order advection
struct SecondOrderCentered <: AbstractAdvectionScheme end
@inline advective_flux(i, N, u, ϕ) = u[i] * (ϕ[i-1] + ϕ[i]) / 2
@inline ∂x_advective_flux(i, N, Δx, u, ϕ) = (advective_flux(i+1, N, u, ϕ) - advective_flux(i, N, u, ϕ)) / Δx

# Convert x to periodic range in [-L/2, L/2] assuming u = 1.
@inline x′(x, t, L) = mod(x + L/2 - t, L) - L/2

# Analytic solutions
@inline ϕ_Gaussian(x, t, L; a=1, c=1/8) = a*exp(-x′(x, t, L)^2 / (2c^2))

function run_advection_experiment(N, L, CFL, ϕ, scheme)
    Δx = L/N
    x = range(-L/2 + Δx/2, L/2 - Δx/2; length=N)
    ϕ₀ = ϕ.(x, 0, L)

    u = OffsetArray(ones(N+2), 0:N+1)
    ϕ₀ = OffsetArray([0, ϕ₀..., 0], 0:N+1)

    create_animation(N, L, x, Δx, CFL, u, ϕ₀, ϕ, nothing)
end

function every(N)
     0 < N <= 16 && return 1
    16 < N <= 32 && return 2
    32 < N <= 64 && return 4
    64 < N       && return 8
end

function create_animation(N, L, x, Δx, CFL, u, ϕ, ϕₐ, scheme)
    T = 2  # two revolutions end time
    Δt = CFL * Δx / maximum(abs, u)
    nt = ceil(T/Δt)

    anim = @animate for iter in 1:nt
        iter % 100 == 0 && @printf("iter = %d/%d\n", iter, nt)

        ϕ[0], ϕ[N+1] = ϕ[N], ϕ[1]  # Fill ghost points.
        for i in 1:N
            ϕ[i] = ϕ[i] - Δt * ∂x_advective_flux(i, N, Δx, u, ϕ)
        end

        title = @sprintf("%s N=%d CFL=%.1f", scheme, N, CFL)
        plot(x, ϕₐ.(x, nt*Δt, L), label="analytic", title=title, xlims=(-0.5, 0.5), ylims=(-0.2, 1.2))
        plot!(x, ϕ[1:N], label="2nd order")
    end every every(N)

    mp4(anim, "tracer_advection_1d.mp4", fps = 30)

    return nothing
end

Ns = [16, 32, 64, 128]
CFLs = [0.1, 0.3, 0.5]
schemes = (SecondOrderCentered,)

run_advection_experiment(16, 1, 0.1, ϕ_Gaussian, nothing)

