using Printf
using OffsetArrays
using DifferentialEquations

ENV["GKSwstype"] = "nul"
using Plots
pyplot()

include("wind_driven_gyres.jl")

const km = 1000
const hour = 3600
const day  = 24hour

#####
##### Advection or flux-reconstruction schemes
#####

abstract type AbstractAdvectionScheme end

struct SecondOrderCentered <: AbstractAdvectionScheme end

@inline advective_flux_x(i, j, u, ϕ, ::SecondOrderCentered) = u[i, j] * (ϕ[i-1, j] + ϕ[i, j]) / 2
@inline advective_flux_y(i, j, v, ϕ, ::SecondOrderCentered) = v[i, j] * (ϕ[i, j-1] + ϕ[i, j]) / 2

@inline ∂x_advective_flux(i, j, Δx, u, ϕ, scheme) =
    (advective_flux_x(i+1, j, u, ϕ, scheme) - advective_flux_x(i, j, u, ϕ, scheme)) / Δx
@inline ∂y_advective_flux(i, j, Δy, v, ϕ, scheme) =
    (advective_flux_y(i, j+1, v, ϕ, scheme) - advective_flux_y(i, j, v, ϕ, scheme)) / Δy

@inline div_advective_flux(i, j, Δx, Δy, u, v, ϕ, scheme) =
    ∂x_advective_flux(i, j, Δx, u, ϕ, scheme) + ∂y_advective_flux(i, j, Δy, v, ϕ, scheme)

#####
##### Right hand side evaluation of the 2D tracer advection equation
#####

function advection!(∂ϕ∂t, ϕ, p, t)
    Nx, Ny, Δx, Hx, Hy, Δy = p.Nx, p.Ny, p.Δx, p.Hx, p.Hy, p.Δy
    u, v, scheme = p.u, p.v, p.scheme

    # Fill ghost points to enforce no-flux boundary conditions.
    ϕ[-Hx+1:0,    :] .= ϕ[1:1,   :]
    ϕ[Nx+1:Nx+Hx, :] .= ϕ[Nx:Nx, :]
    ϕ[:,    -Hy+1:0] .= ϕ[:,   1:1]
    ϕ[:, Ny+1:Ny+Hy] .= ϕ[:, Ny:Ny]

    for j in 1:Ny, i in 1:Nx
        ∂ϕ∂t[i, j] = -div_advective_flux(i, j, Δx, Δy, u, v, ϕ, scheme)
    end
end

#####
##### Initial (= final) conditions
#####

@inline ϕ_Gaussian(x, y; L, A, σˣ, σʸ) = A * exp(-(x-L/2)^2/(2σˣ^2) -(y-L/2)^2/(2σʸ^2))
@inline ϕ_Square(x, y; L, A, σˣ, σʸ)   = A * (-σˣ <= x-L/2 <= σˣ) * (-σʸ <= y-L/2 <= σʸ)

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_problem(Nx, Ny, T, CFL, ϕₐ, time_stepper, scheme;
                       u, v, L=4000km, τ₀=1, β=1e-11, r=0.04*β*L,
                       A=1, σˣ=10km, σʸ=10km)
    Δx = L/Nx
    Δy = L/Ny
    Hx = Hy = 3

    xC = range(Δx/2, L - Δx/2, length=Nx)
    yC = range(Δy/2, L - Δy/2, length=Ny)

    xF = range(0, L, length=Nx+1)
    yF = range(0, L, length=Ny+1)
    
    u = OffsetArray(zeros(Nx+2Hx, Ny+2Hy), -Hx+1:Nx+Hx, -Hy+1:Ny+Hy)
    v = OffsetArray(zeros(Nx+2Hx, Ny+2Hy), -Hx+1:Nx+Hx, -Hy+1:Ny+Hy)
    ϕ = OffsetArray(zeros(Nx+2Hx, Ny+2Hy), -Hx+1:Nx+Hx, -Hy+1:Ny+Hy)

    for j in 1:Ny, i in 1:Nx
        u[i, j] = u_Stommel(xF[i], yF[j], L=L, τ₀=τ₀, β=β, r=r)
        v[i, j] = v_Stommel(xF[i], yF[j], L=L, τ₀=τ₀, β=β, r=r)
        ϕ[i, j] = ϕₐ(xC[i], yC[j], L=L, A=A, σˣ=σˣ, σʸ=σʸ)
    end

    U_max = max(maximum(abs, u), maximum(abs, v))
    u = u ./ U_max
    v = v ./ U_max

    Δt = CFL * min(Δx, Δy) / max(maximum(abs, u), maximum(abs, v))
    
    @info @sprintf("Nx=%d, Ny=%d, L=%.3f km, Δx=%.3f km, Δy=%.3f km, Δt=%.3f hours",
                   Nx, Ny, L/km, Δx/km, Δy/km, Δt/hour)

    tspan = (0.0, T)
    params = (Nx=Nx, Ny=Ny, Hx=Hx, Hy=Hy, Δx=Δx, Δy=Δy, u=u, v=v, scheme=scheme)
    return xC, yC, Δt, ODEProblem(advection!, ϕ, tspan, params)
end

function create_animation(Nx, Ny, T, CFL, ϕₐ, time_stepper, scheme;
                          u, v, L=1000km, τ₀=1, β=1e-11, r=0.04*β*L,
                          A=1, σˣ=50km, σʸ=50km)

    xC, yC, Δt, prob = setup_problem(Nx, Ny, T, CFL, ϕₐ, time_stepper, scheme,
                                    u=u, v=v, L=L, τ₀=τ₀, β=β, r=r, A=A, σˣ=σˣ, σʸ=σʸ)

    integrator = init(prob, time_stepper, adaptive=false, dt=Δt)
    nt = ceil(Int, T/Δt)

    function every(n)
          0 < n <= 128 && return 1
        128 < n <= 256 && return 2
        256 < n <= 512 && return 4
        512 < n        && return 8
    end

    anim = @animate for iter in 0:nt-1
        iter % 100 == 0 && @info @sprintf("iter = %d/%d\n", iter, nt)

        step!(integrator)

        title = @sprintf("%s %s N=%d CFL=%.2f", typeof(scheme), typeof(time_stepper), Nx, CFL)
        contourf(xC ./ L, yC ./ L, reverse(transpose(integrator.u[1:Nx, 1:Ny]), dims=1),
                 title=title, xlabel="x/L", ylabel="y/L",
                 xlims=(0, 1), ylims=(0, 1),
                 levels=20, fill=:true, color=:balance, clims=(-1.0, 1.0), legend=false,
                 aspect_ratio=:equal, dpi=200)  

    end every every(nt)

    anim_filename = @sprintf("%s_%s_%s_N%d_CFL%.2f.mp4", ic_name(ϕₐ), typeof(scheme), typeof(time_stepper), Nx, CFL)
    mp4(anim, anim_filename, fps = 15)

    return nothing
end

Nx = Ny = 32
L = 1000km
T = 365day
CFL = 0.3
create_animation(Nx, Ny, T, CFL, ϕ_Gaussian, Tsit5(), SecondOrderCentered(), u=u_Stommel, v=v_Stommel)

