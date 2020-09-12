using Printf
using Logging
using Plots

using Oceananigans
using Oceananigans.Advection

using Oceananigans.Grids: xnodes

ENV["GKSwstype"] = "100"

Logging.global_logger(OceananigansLogger())

#####
##### Initial conditions and analytic solutions
#####

# Convert x to periodic range in [-L/2, L/2] assuming u = 1.
@inline x′(x, t, L) = mod(x + L/2 - t, L) - L/2

# Analytic solutions
@inline ϕ_Gaussian(x, t; L, a=1, c=1/8) = a * exp(-x′(x, t, L)^2 / (2c^2))
@inline ϕ_Square(x, t; L, w=0.15) = -w <= x′(x, t, L) <= w ? 1.0 : 0.0

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_model(N, L, U, ϕₐ, advection_scheme)
    topology = (Periodic, Flat, Flat)
    domain = (x=(-L/2, L/2), y=(0, 1), z=(0, 1))
    grid = RegularCartesianGrid(topology=topology, size=(N, 1, 1), halo=(3, 3, 3); domain...)

    model = IncompressibleModel(
             grid = grid,
        advection = advection_scheme,
          tracers = :c,
         buoyancy = nothing,
          closure = IsotropicDiffusivity(ν=0, κ=0)
    )

    set!(model, u = U, c = (x, y, z) -> ϕₐ(x, 0; L=L))

    return model
end

function create_animation(N, L, CFL, ϕₐ, advection_scheme; U=1.0, T=2.0)
    model = setup_model(N, L, U, ϕₐ, advection_scheme)
    
    c = model.tracers.c
    x = xnodes(c)
    Δt = CFL * model.grid.Δx / U
    Nt = ceil(Int, T/Δt)

    function every(n)
          0 < n <= 128 && return 1
        128 < n <= 256 && return 2
        256 < n <= 512 && return 4
        512 < n        && return 8
    end

    anim_filename = @sprintf("%s_%s_N%d_CFL%.2f.mp4", ic_name(ϕₐ), typeof(advection_scheme), N, CFL)

    anim = @animate for iter in 1:Nt
        iter % 10 == 0 && @info "$anim_filename, iter = $iter/$Nt"

        time_step!(model, Δt, euler = iter == 1)

        ϕ_analytic = ϕₐ.(x, model.clock.time; L=L)

        title = @sprintf("%s N=%d CFL=%.2f", typeof(advection_scheme), N, CFL)
        plot(x, ϕ_analytic, lw=2, label="analytic", title=title, xlims=(-L/2, L/2), ylims=(-0.2, 1.2), dpi=200)
        plot!(x, interior(c)[:], lw=2, label="Oceananigans")
    end every every(Nt)

    mp4(anim, anim_filename, fps = 15)

    return model
end

#####
##### Run 1D tracer advection experiments
#####

L = 1
ϕs = (ϕ_Gaussian, ϕ_Square)
advection_schemes = (CenteredSecondOrder(), CenteredFourthOrder(), WENO5())
Ns = [16, 32, 64]
CFLs = (0.2, 0.5, 1.0)

for ϕ in ϕs, scheme in advection_schemes, N in Ns, CFL in CFLs
    @info @sprintf("Creating two-revolution animation [%s, %s, N=%d, CFL=%.2f]...", ic_name(ϕ), typeof(advection_scheme), N, CFL)
    create_animation(N, L, CFL, ϕ, scheme)
end
