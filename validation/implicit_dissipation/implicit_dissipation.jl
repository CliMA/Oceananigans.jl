using Oceananigans
using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.Models: VarianceDissipationComputation
using Printf
using GLMakie

# the initial condition
@inline G(x, β, z) = exp(-β*(x - z)^2)
@inline F(x, α, a) = √(max(1 - α^2*(x-a)^2, 0.0))

const Z = -0.7
const δ = 0.005
const β = log(2)/(36*δ^2)
const a = 0.5
const α = 10

@inline function bᵢ(x) 
    if x <= -0.6 && x >= -0.8
        return 1/6*(G(x, β, Z-δ) + 4*G(x, β, Z) + G(x, β, Z+δ))
    elseif x <= -0.2 && x >= -0.4
        return 1.0
    elseif x <= 0.2 && x >= 0.0
        return 1.0 - abs(10 * (x - 0.1))
    elseif x <= 0.6 && x >= 0.4
        return 1/6*(F(x, α, a-δ) + 4*F(x, α, a) + F(x, α, a+δ))
    else
        return 0.0
    end
end


grid = RectilinearGrid(size = N, halo = 6, x = (-1, 1), topology = (Periodic, Flat, Flat))

model = NonhydrostaticModel(; grid, tracers = :b, timestepper = :QuasiAdamsBashforth2, advection) 

set!(model.tracers.b, bᵢ)
set!(model.velocities.u, 1.0)

b₀ = deepcopy(model.tracers.b)
Δt = 0.1 * minimum_xspacing(grid)

dissipation_computation = VarianceDissipationComputation(model)
simulation  = Simulation(model; Δt, stop_time = 10)
simulation.callbacks[:compute_dissipation] = Callback(dissipation_computation, IterationInterval(1))

outputs = (; Px = dissipation_computation.production.b.x, b = model.tracers.b)

# Save the dissipation
simulation.output_writers[:dissipation] = JLD2OutputWriter(model, dissipation_computation.production.b,
                                                            filename = "dissipation.jld2", 
                                                            schedule = IterationInterval(10))

run!(simulation)
b = FieldTimesSeries("dissipation.jld2", "b")
P = FieldTimesSeries("dissipation.jld2", "Px")

iter = Observable(1)

bn = @lift(interior(b[$iter], :, 1, 1))
Pn = @lift(interior(b[$iter], :, 1, 1))

fig = Figure(size = (800, 400))
ax  = Axis(fig[1, 1], xlabel = L"x", ylabel = L"tracer")
lines!(ax, xnodes(b[1]), b0, color = :grey, linestyle = :dash, linewidth = 2)
lines!(ax, xnodes(b[1]), bn, color = :blue)

ax  = Axis(fig[1, 1], xlabel = L"x", ylabel = L"variance dissipation")
lines!(ax, xnodes(b[1]), Pn, color = :red)

record(fig, "implicit_dissipation.mp4", 1:length(b), framerate=8) do i
    n[] = i
end