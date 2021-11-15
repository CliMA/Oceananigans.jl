using Printf
using Logging
using Plots

using Oceananigans
using Oceananigans.Advection
using Oceananigans.TimeSteppers

using Oceananigans.Grids: xnodes

ENV["GKSwstype"] = "100"

#####
##### Initial conditions and analytic solutions
#####

# Convert x to periodic range in [-L/2, L/2] assuming u = 1.
@inline x′(x, t, L, U) = mod(x + L/2 - U * t, L) - L/2

# Analytic solutions
@inline ϕ_Gaussian(x, t; L, U, a=1, c=1/8) = a * exp(-x′(x, t, L, U)^2 / (2c^2))
@inline ϕ_Square(x, t; L, U, w=0.15) = -w <= x′(x, t, L, U) <= w ? 1.0 : 0.0

ic_name(::typeof(ϕ_Gaussian)) = "Gaussian"
ic_name(::typeof(ϕ_Square))   = "Square"

#####
##### Experiment functions
#####

function setup_model(N, L, U, ϕₐ, time_stepper, advection_scheme)
    topology = (Periodic, Flat, Flat)
    grid = RectilinearGrid(topology=topology, size=(N, ), halo=(9, ), x=(-L/2, L/2))

    model = NonhydrostaticModel(
               grid = grid,
        timestepper = time_stepper,
          advection = advection_scheme,
            tracers = :c,
           buoyancy = nothing,
            closure = IsotropicDiffusivity(ν=0, κ=0)
    )

    set!(model, u = U, v = (x, y, z) -> ϕₐ(x, 0; L=L, U=U), c = (x, y, z) -> ϕₐ(x, 0; L=L, U=U))

    return model
end

function short_name(ts)
    ts == :QuasiAdamsBashforth2 && return "QAB2"
    ts == :RungeKutta3 && return "RK3"
end

function create_animation(N, L, CFL, ϕₐ, time_stepper, advection_scheme; U=1.0, T=2.0)
    model = setup_model(N, L, U, ϕₐ, time_stepper, advection_scheme)
    
    v, c = model.velocities.v, model.tracers.c
    x = xnodes(c)
    Δt = CFL * model.grid.Δxᶜᵃᵃ / abs(U)
    Nt = ceil(Int, T/Δt)

    function every(n)
          0 < n <= 128 && return 1
        128 < n <= 256 && return 2
        256 < n <= 512 && return 4
        512 < n        && return 8
    end

    scheme_name = typeof(advection_scheme)

    anim_filename = @sprintf("%s_%s_%s_N%d_CFL%.2f_U%+d.gif", ic_name(ϕₐ), time_stepper, scheme_name, N, CFL, U)

    anim = @animate for iter in 0:Nt
        iter % 10 == 0 && @info "$anim_filename, iter = $iter/$Nt"

        ϕ_analytic = ϕₐ.(x, model.clock.time; L=L, U=U)

        title = @sprintf("%s %s N=%d CFL=%.2f", short_name(time_stepper), scheme_name, N, CFL)
        plot(x, ϕ_analytic, lw=2, label="analytic", title=title, xlims=(-L/2, L/2), ylims=(-0.2, 1.2), dpi=200)
        plot!(x, interior(v)[:], ls=:solid, lw=2, label="Oceananigans v")
        plot!(x, interior(c)[:], ls=:dot,   lw=2, label="Oceananigans c")

        if time_stepper == :QuasiAdamsBashforth2
            time_step!(model, Δt, euler = iter == 0)
        else
            time_step!(model, Δt)
        end
    end every every(Nt)

    gif(anim, anim_filename, fps=15)

    return model
end

#####
##### Run 1D periodic advection experiments
#####

L = 1
ϕs = (ϕ_Gaussian, ϕ_Square)
time_steppers = (:RungeKutta3,)
advection_schemes = (CenteredSecondOrder(), CenteredFourthOrder(), UpwindBiasedThirdOrder(), UpwindBiasedFifthOrder(), WENO5())
Ns = [16, 64]
CFLs = (0.5, 1.7)
Us = [+1, -1]

for ϕ in ϕs, ts in time_steppers, scheme in advection_schemes, N in Ns, CFL in CFLs, U in Us
    scheme_name = typeof(scheme)
    @info @sprintf("Creating two-revolution animation [%s, %s, %s, N=%d, CFL=%.2f, U=%+d]...", ic_name(ϕ), ts, scheme_name, N, CFL, U)
    create_animation(N, L, CFL, ϕ, ts, scheme, U=U)
end
