using Printf
using OffsetArrays
using DifferentialEquations
using Plots

include("wind_driven_gyres.jl")

const km = 1000
const day = 3600*24

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

@inline ϕ_Gaussian(x, y; A, σˣ, σʸ) = A * exp(-x^2/(2σˣ^2) -y^2/(2σʸ^2))
@inline ϕ_Square(x, y; A, σˣ, σʸ)   = A * (-σˣ <= x <= σˣ) * (-σʸ <= y <= σʸ)

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_problem(Nx, Ny, T, CFL, ϕₐ, time_stepper, scheme; u, v,
                       L=4000km, τ₀=1, β=1e-11, r=0.04*β*L, A=1, σˣ=10km, σʸ=10km)
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
        ϕ[i, j] = ϕₐ(xC[i], yC[j], A=A, σˣ=σˣ, σʸ=σʸ)
    end

    @show Δx, Δy
    @show maximum(abs, u)
    @show maximum(abs, v)
    u = u ./ maximum(abs, u)
    v = v ./ maximum(abs, v)
    Δt = CFL * min(Δx, Δy) / max(maximum(abs, u), maximum(abs, v))
    @show Δt

    tspan = (0.0, T)
    params = (Nx=Nx, Ny=Ny, Hx=Hx, Hy=Hy, Δx=Δx, Δy=Δy, u=u, v=v, scheme=scheme)
    return xC, yC, Δt, ODEProblem(advection!, ϕ, tspan, params)
end

Nx = Ny = 32
L = 1000km
T = 1day
CFL = 0.1
x, y, Δt, prob = setup_problem(Nx, Ny, T, CFL, ϕ_Gaussian, Tsit5(), SecondOrderCentered(),
                              u=u_Stommel, v=v_Stommel)

integrator = init(prob, Tsit5(), adaptive=false, dt=Δt)
step!(integrator)

