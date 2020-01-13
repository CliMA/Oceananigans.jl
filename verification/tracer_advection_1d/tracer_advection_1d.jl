using Printf
using OffsetArrays
using DifferentialEquations
using Plots

#####
##### Advection or flux-reconstruction schemes
#####

abstract type AbstractAdvectionScheme end

struct FirstOrderUpwind <: AbstractAdvectionScheme end
@inline ∂x_advective_flux(i, Δx, u, ϕ, ::FirstOrderUpwind) =
    max(u[i], 0) * (ϕ[i] - ϕ[i-1])/Δx + min(u[i], 0) * (ϕ[i+1] - ϕ[i])/Δx

struct SecondOrderCentered <: AbstractAdvectionScheme end
@inline advective_flux(i, u, ϕ, ::SecondOrderCentered) = u[i] * (ϕ[i-1] + ϕ[i]) / 2
@inline ∂x_advective_flux(i, Δx, u, ϕ, scheme) =
    (advective_flux(i+1, u, ϕ, scheme) - advective_flux(i, u, ϕ, scheme)) / Δx

include("weno.jl")
struct WENO5 end
@inline ∂x_advective_flux(i, Δx, u, ϕ, ::WENO5) = u[i] * (weno5_flux(i+1, ϕ) - weno5_flux(i, ϕ)) / Δx

#####
##### Right hand side evaluation of the advection equation
#####

function advection!(∂ϕ∂t, ϕ, p, t)
    N, H = p.N, p.H
    ϕ[-H+1:0], ϕ[N+1:N+H] = ϕ[N-H+1:N], ϕ[1:H] # Fill ghost points to enforce periodic boundary conditions.
    for i in 1:N
        ∂ϕ∂t[i] = -∂x_advective_flux(i, p.Δx, p.u, ϕ, p.scheme)
    end
end

#####
##### Initial conditions and analytic solutions
#####

# Convert x to periodic range in [-L/2, L/2] assuming u = 1.
@inline x′(x, t, L) = mod(x + L/2 - t, L) - L/2

# Analytic solutions
@inline ϕ_Gaussian(x, t, L; a=1, c=1/8) = a*exp(-x′(x, t, L)^2 / (2c^2))
@inline ϕ_Square(x, t, L; w=0.15) = -w <= x′(x, t, L) <= w ? 1.0 : 0.0

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_problem(N, L, T, CFL, ϕₐ, time_stepper, scheme)
    Δx = L/N
    x = range(-L/2 + Δx/2, L/2 - Δx/2; length=N)
    ϕ₀ = ϕₐ.(x, 0, L)

    H = 3
    halo = ones(H)
    u  = [halo..., ones(N)..., halo...]
    ϕ₀ = [halo..., ϕ₀...,      halo...]
    u  = OffsetArray(u,  -H+1:N+H)
    ϕ₀ = OffsetArray(ϕ₀, -H+1:N+H)

    Δt = CFL * Δx / maximum(abs, u)

    tspan = (0.0, T)
    params = (N=N, H=H, Δx=Δx, u=u, scheme=scheme)
    return x, Δt, ODEProblem(advection!, ϕ₀, tspan, params)
end

function create_figure(N, L, CFL, ϕₐ, time_stepper, scheme; T=1.0)
    x, Δt, prob = setup_problem(N, L, T, CFL, ϕₐ, time_stepper, scheme)
    sol = solve(prob, time_stepper, adaptive=false, dt=Δt)
    ϕ = sol[:, end]
    @info "Solver return code: $(sol.retcode)"

    title = @sprintf("%s %s N=%d CFL=%.2f", typeof(scheme), typeof(time_stepper), N, CFL)
    ϕ_analytic = ϕₐ.(x, T, L)
    p = plot(x, ϕ_analytic, label="analytic", title=title, xlims=(-0.5, 0.5), ylims=(-0.2, 1.2), dpi=200)
    plot!(p, x, ϕ[1:N], label="numerical")

    fig_filename = @sprintf("%s_%s_%s_N%d_CFL%.2f.png", ic_name(ϕₐ), typeof(scheme), typeof(time_stepper), N, CFL)
    savefig(p, fig_filename)

    return nothing
end

function create_animation(N, L, CFL, ϕₐ, time_stepper, scheme; T=2.0)
    x, Δt, prob = setup_problem(N, L, T, CFL, ϕₐ, time_stepper, scheme)
    integrator = init(prob, time_stepper, adaptive=false, dt=Δt)
    nt = ceil(Int, T/Δt)

    function every(n)
          0 < n <= 128 && return 1
        128 < n <= 256 && return 2
        256 < n <= 512 && return 4
        512 < n        && return 8
    end

    anim = @animate for iter in 1:nt
        iter % 10 == 0 && @info @sprintf("iter = %d/%d\n", iter, nt)

        step!(integrator)

        title = @sprintf("%s %s N=%d CFL=%.2f", typeof(scheme), typeof(time_stepper), N, CFL)
        ϕ_analytic = ϕₐ.(x, integrator.t, L)
        plot(x, ϕ_analytic, lw=2, label="analytic", title=title, xlims=(-0.5, 0.5), ylims=(-0.2, 1.2), dpi=200)
        plot!(x, integrator.u[1:N], lw=2, label="numerical")
    end every every(nt)

    anim_filename = @sprintf("%s_%s_%s_N%d_CFL%.2f.mp4", ic_name(ϕₐ), typeof(scheme), typeof(time_stepper), N, CFL)
    mp4(anim, anim_filename, fps = 15)

    return nothing
end

#####
##### Run 1D tracer advection experiments
#####

L = 1
ϕs = (ϕ_Gaussian,)
time_steppers = (AB3(), CarpenterKennedy2N54())
schemes = (SecondOrderCentered(), WENO5())
Ns = [16, 32, 64, 128]
CFLs = Dict(
    Euler => (0.05, 0.3, 0.5),
    AB3   => (0.05, 0.3, 0.5, 0.9),
    CarpenterKennedy2N54 => (0.05, 0.3, 0.5, 0.9, 1.5, 2.0, 3.0, 4.0)
)

for ϕ in ϕs, ts in time_steppers, scheme in schemes, N in Ns, CFL in CFLs[typeof(ts)]
    @info @sprintf("Creating one-revolution figure [%s, %s, %s, N=%d, CFL=%.2f]...", ic_name(ϕ), typeof(ts), typeof(scheme), N, CFL)
    create_figure(N, L, CFL, ϕ, ts, scheme)

    # @info @sprintf("Creating two-revolution animation [%s, N=%d, CFL=%.2f]...", typeof(scheme), N, CFL)
    # create_animation(N, L, CFL, ϕ, scheme)
end

create_animation(64, L, 0.6, ϕ_Gaussian, AB3(), SecondOrderCentered())
create_animation(32, L, 1.8, ϕ_Gaussian, CarpenterKennedy2N54(), WENO5())

create_animation(64, L, 0.6, ϕ_Square, AB3(), SecondOrderCentered())
create_animation(32, L, 1.8, ϕ_Square, CarpenterKennedy2N54(), WENO5())
create_animation(256, L, 1.8, ϕ_Square, CarpenterKennedy2N54(), WENO5())

