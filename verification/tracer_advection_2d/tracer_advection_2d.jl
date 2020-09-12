using Printf
using Logging
using Plots

using Oceananigans
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Utils

using Oceananigans.Grids: ynodes, znodes

include("stommel_gyre.jl")

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

function setup_simulation(N, T, CFL, ϕₐ, advection_scheme; u, v)
    topology = (Flat, Bounded, Bounded)
    domain = (x=(0, 1), y=(0, L), z=(0, L))
    grid = RegularCartesianGrid(topology=topology, size=(1, N, N), halo=(3, 3, 3); domain...)

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

    simulation = Simulation(model, Δt=Δt, stop_time=T, progress=print_progress, iteration_interval=10)

    filename = @sprintf("stommel_gyre_%s_%s_N%d_CFL%.2f.nc", ic_name(ϕₐ), typeof(advection_scheme), N, CFL)
    fields = Dict("v" => model.velocities.v, "w" => model.velocities.w, "c" => model.tracers.c)
    output_attributes = Dict("c" => Dict("longname" => "passive tracer"))

    simulation.output_writers[:fields] =
        NetCDFOutputWriter(model, fields, filename=filename, time_interval=0.01,
                           output_attributes=output_attributes, verbose=true)

    return simulation
end

function print_progress(simulation)
    model = simulation.model

    progress = 100 * (model.clock.time / simulation.stop_time)

    v_max = maximum(abs, interior(model.velocities.v))
    w_max = maximum(abs, interior(model.velocities.w))
    c_min, c_max = extrema(interior(model.tracers.c))

    i, t = model.clock.iteration, model.clock.time
    @info @sprintf("[%06.2f%%] i: %d, t: %.4f, U_max: (%.2e, %.2e), c: (min=%.3e, max=%.3e)",
                   progress, i, t, v_max, w_max, c_min, c_max)
end

function create_animation(N, T, CFL, ϕₐ, scheme; u, v)
    model, Δt = setup_model(N, ϕₐ, scheme, u=u, v=v)

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

    anim_filename = @sprintf("%s_%s_N%d_CFL%.2f.mp4", ic_name(ϕₐ), typeof(scheme), N, CFL)

    anim = @animate for iter in 0:Nt-1
        @info "$anim_filename: iter = $(model.clock.iteration)/$Nt"

        time_step!(model, Δt, euler = model.clock.iteration == 0)

        # if !reverse_uv && model.clock.time >= T/2
        #     set!(model, v = (x, y, z) -> -u(x, y, z), w = (x, y, z) -> -v(x, y, z))
        #     reverse_uv = true
        # end

        title = @sprintf("%s N=%d CFL=%.2f", typeof(scheme), N, CFL)

        contourf(y ./ L, z ./ L, dropdims(interior(c), dims=1),
                 title=title, xlabel="x/L", ylabel="y/L",
                 xlims=(0, 1), ylims=(0, 1),
                 levels=20, fill=:true, color=:balance, clims=(-1.0, 1.0),
                 aspect_ratio=:equal, dpi=300, show= iter % 100 == 0)

    end every 100

    mp4(anim, anim_filename, fps = 15)

    return model
end

L = τ₀ = β = 1
r = 0.04β*L

A  = 1
σˣ = σʸ = 0.05L

N = 64
CFL = 0.2
T = 1

u = (x, y, z) -> u_Stommel(y, z, L=L, τ₀=τ₀, β=β, r=r)
v = (x, y, z) -> v_Stommel(y, z, L=L, τ₀=τ₀, β=β, r=r)
ϕ = (x, y, z) -> ϕ_Gaussian(y, z, L=L, A=A, σˣ=σˣ, σʸ=σʸ)

ic_name(::typeof(ϕ)) = ic_name(ϕ_Gaussian)

# model, Δt = create_animation(N, T, CFL, ϕ, CenteredSecondOrder(), u=u, v=v)
