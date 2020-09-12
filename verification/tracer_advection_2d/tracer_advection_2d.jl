using Printf
using Logging
using Plots

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Utils

using Oceananigans.Grids: ynodes, znodes

include("stommel_gyre.jl")

const km = kilometer

ENV["GKSwstype"] = "100"
pyplot()

Logging.global_logger(OceananigansLogger())

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

function setup_model(Nx, Ny, ϕₐ, advection_scheme; u, v)
    topology = (Flat, Bounded, Bounded)
    domain = (x=(0, 1), y=(0, L), z=(0, L))
    grid = RegularCartesianGrid(topology=topology, size=(1, Nx, Ny), halo=(3, 3, 3); domain...)

    model = IncompressibleModel(
             grid = grid,
        advection = advection_scheme,
          tracers = :c,
         buoyancy = nothing,
          closure = IsotropicDiffusivity(ν=0, κ=0)
    )

    set!(model, v=u, w=v, c=ϕₐ)

    v_max = maximum(abs, interior(model.velocities.v))
    w_max = maximum(abs, interior(model.velocities.w))
    Δt = CFL * min(grid.Δy, grid.Δz) / max(v_max, w_max)

    @info @sprintf("Nx=%d, Ny=%d, L=%.3f km, Δx=%.3f km, Δy=%.3f km, Δt=%.3f hours",
                   Nx, Ny, L/km, grid.Δx/km, grid.Δy/km, Δt/hour)

    return model, Δt
end

function create_animation(Nx, Ny, T, CFL, ϕₐ, scheme; u, v)
    model, Δt = setup_model(Nx, Ny, ϕₐ, scheme, u=u, v=v)

    c = model.tracers.c
    y, z = ynodes(c), znodes(c)
    Nt = ceil(Int, T/Δt)

    function every(n)
          0 < n <= 128 && return 1
        128 < n <= 256 && return 2
        256 < n <= 512 && return 4
        512 < n        && return 8
    end

    reverse_uv = false

    anim_filename = @sprintf("%s_%s_N%d_CFL%.2f.mp4", ic_name(ϕₐ), typeof(scheme), Nx, CFL)

    anim = @animate for iter in 0:Nt-1
        iter % 10 == 0 && @info "$anim_filename: iter = $iter/$Nt"

        time_step!(model, Δt, euler = iter == 0)

        if reverse_uv || model.clock.time >= T/2
            set!(model, v = (x, y, z) -> -u(x, y, z), w = (x, y, z) -> -v(x, y, z))
            reverse_uv = true
        end

        title = @sprintf("%s N=%d CFL=%.2f", typeof(scheme), Nx, CFL)

        contourf(y ./ L, z ./ L, dropdims(interior(c), dims=1),
                 title=title, xlabel="x/L", ylabel="y/L",
                 xlims=(0, 1), ylims=(0, 1),
                 levels=20, fill=:true, color=:balance, clims=(-1.0, 1.0),
                 aspect_ratio=:equal, dpi=300)

    end every every(100)

    mp4(anim, anim_filename, fps = 60)

    return nothing
end

L  = 1000km
τ₀ = 1
β  = 1e-11
r  = 0.04*β*L

A  = 1
σˣ = 50km
σʸ = 50km

T = 100day
Nx = Ny = 64
CFL = 0.2

u = (x, y, z) -> u_Stommel(y, z, L=L, τ₀=τ₀, β=β, r=r) / u_Stommel_max(L=L, τ₀=τ₀, β=β)
v = (x, y, z) -> v_Stommel(y, z, L=L, τ₀=τ₀, β=β, r=r) / v_Stommel_max(L=L, τ₀=τ₀, β=β)
ϕ = (x, y, z) -> ϕ_Gaussian(y, z, L=L, A=A, σˣ=σˣ, σʸ=σʸ)

ic_name(::typeof(ϕ)) = ic_name(ϕ_Gaussian)

create_animation(Nx, Ny, T, CFL, ϕ, CenteredSecondOrder(), u=u, v=v)
