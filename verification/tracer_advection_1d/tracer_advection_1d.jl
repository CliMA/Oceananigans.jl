using Printf
using OffsetArrays
using Plots

abstract type AbstractAdvectionScheme end

# Second order advection
struct SecondOrderCentered <: AbstractAdvectionScheme end
@inline advective_flux(i, N, u, ϕ) = u[i] * (ϕ[i-1] + ϕ[i]) / 2
@inline ∂x_advective_flux(i, N, Δx, u, ϕ) = (advective_flux(i+1, N, u, ϕ) - advective_flux(i, N, u, ϕ)) / Δx

function time_step_advection!(ϕ, u, N, Δx, Δt, scheme)
    ϕ[0], ϕ[N+1] = ϕ[N], ϕ[1]  # Fill ghost points to enforce periodic boundary conditions.
    for i in 1:N
        ϕ[i] = ϕ[i] - Δt * ∂x_advective_flux(i, N, Δx, u, ϕ)
    end
end

# Convert x to periodic range in [-L/2, L/2] assuming u = 1.
@inline x′(x, t, L) = mod(x + L/2 - t, L) - L/2

# Analytic solutions
@inline ϕ_Gaussian(x, t, L; a=1, c=1/8) = a*exp(-x′(x, t, L)^2 / (2c^2))

function create_figure(N, L, CFL, ϕₐ, scheme)
    Δx = L/N
    x = range(-L/2 + Δx/2, L/2 - Δx/2; length=N)
    ϕ₀ = ϕₐ.(x, 0, L)

    u = OffsetArray(ones(N+2), 0:N+1)
    ϕ = OffsetArray([0, ϕ₀..., 0], 0:N+1)

    T = 1  # one revolution end time
    Δt = CFL * Δx / maximum(abs, u)
    nt = ceil(T/Δt)

    for _ in 1:nt
        time_step_advection!(ϕ, u, N, Δx, Δt, scheme)
    end

    title = @sprintf("%s N=%d CFL=%.1f", typeof(scheme), N, CFL)
    ϕ_analytic = ϕₐ.(x, T, L)
    p = plot(x, ϕ_analytic, label="analytic", title=title, xlims=(-0.5, 0.5), ylims=(-0.2, 1.2), dpi=200)
    plot!(p, x, ϕ[1:N], label="$(typeof(scheme))")

    fig_filename = @sprintf("%s_N%d_CFL%.1f.png", typeof(scheme), N, CFL)
    savefig(p, fig_filename)

    return nothing
end

function create_animation(N, L, CFL, ϕₐ, scheme)
    Δx = L/N
    x = range(-L/2 + Δx/2, L/2 - Δx/2; length=N)
    ϕ₀ = ϕₐ.(x, 0, L)

    u = OffsetArray(ones(N+2), 0:N+1)
    ϕ = OffsetArray([0, ϕ₀..., 0], 0:N+1)

    T = 2  # two revolutions end time
    Δt = CFL * Δx / maximum(abs, u)
    nt = ceil(T/Δt)

    function every(N)
         0 < N <= 16 && return 10
        16 < N <= 32 && return 2
        32 < N <= 64 && return 4
        64 < N       && return 8
    end

    anim = @animate for iter in 1:nt
        iter % 100 == 0 && @printf("iter = %d/%d\n", iter, nt)

        time_step_advection!(ϕ, u, N, Δx, Δt, scheme)

        title = @sprintf("%s N=%d CFL=%.1f", typeof(scheme), N, CFL)
        ϕ_analytic = ϕₐ.(x, nt*Δt, L)
        plot(x, ϕ_analytic, label="analytic", title=title, xlims=(-0.5, 0.5), ylims=(-0.2, 1.2), dpi=200)
        plot!(x, ϕ[1:N], label="$(typeof(scheme))")
    end every every(N)

    anim_filename = @sprintf("%s_N%d_CFL%.1f.mp4", typeof(scheme), N, CFL)
    mp4(anim, anim_filename, fps = 30)

    return nothing
end

L = 1
Ns = [16, 32, 64, 128]
CFLs = [0.1, 0.3, 0.5]
schemes = (SecondOrderCentered(),)

for N in Ns, CFL in CFLs, scheme in schemes
    @info @sprintf("Creating one-revolution figure [%s, N=%d, CFL=%.1f]...", typeof(scheme), N, CFL)
    create_figure(N, L, CFL, ϕ_Gaussian, scheme)
    
    # @info @sprintf("Creating two-revolution animation [%s, N=%d, CFL=%.1f]...", typeof(scheme), N, CFL)
    # create_animation(N, L, CFL, ϕ, scheme)
end
