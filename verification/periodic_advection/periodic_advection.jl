using Printf
using Plots
using JULES
using Oceananigans

using Oceananigans.Grids: Cell, xnodes

ENV["GKSwstype"] = "100"

N = 64
L = 1
T = 1
Δt = 1e-2
Nt = Int(T/Δt)

topo = (Periodic, Periodic, Periodic)
domain = (x=(-L/2, L/2), y=(0, 1), z=(0, 1))
grid = RegularCartesianGrid(topology=topo, size=(N, 1, 1), halo=(2, 2, 2); domain...)

model = CompressibleModel(
                      grid = grid,
                     gases = DryEarth(),
    thermodynamic_variable = Entropy(),
                   closure = IsotropicDiffusivity(ν=0, κ=0),
                   gravity = 0.0,
               tracernames = (:ρ, :ρs, :ρc)
)

gas = model.gases.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
u₀₀, T₀₀, ρ₀₀, s₀₀ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀

ρ₀(x, y, z) = 1
p₀(x, y, z) = 1
T₀(x, y, z) = p₀(x, y, z) / (R*ρ₀(x, y, z))
ρs₀(x, y, z) = ρ₀(x, y, z) * (s₀₀ + cᵥ * log(T₀(x, y, z)/T₀₀) - R * log(ρ₀(x, y, z)/ρ₀₀))

set!(model.tracers.ρ, ρ₀)
set!(model.momenta.ρu, 1)
set!(model.tracers.ρs, ρs₀)
update_total_density!(model)

@inline x′(x, t, L, U) = mod(x + L/2 - U * t, L) - L/2
@inline ϕ_Gaussian(x, t; L, U, a=1, c=1/8) = a * exp(-x′(x, t, L, U)^2 / (2c^2))
ρc₀(x, y, z) = ϕ_Gaussian(x, 0, L=L, U=1)
set!(model.tracers.ρc, ρc₀)

anim = @animate for n in 1:Nt
    @info "iteration $n/$Nt"
    time_step!(model, Δt)

    title = @sprintf("Periodic advection t=%.3f", model.clock.time)

    x = xnodes(Cell, grid)
    ρc = interior(model.tracers.ρc)[:]
    plot(x, ρc, lw=2, label="", title=title, xlims=(-1, 1), ylims=(0, 1), dpi=200)
end

mp4(anim, "periodic_advection.mp4", fps=15)
