using Oceananigans
using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels: PrescribedVelocityFields
using Oceananigans.Models: TracerVarianceDissipation
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

function save_previous_buoyancy!(simulation)
    bⁿ⁻¹ = simulation.model.auxiliary_fields.bⁿ⁻¹
    bⁿ   = simulation.model.tracers.b

    parent(bⁿ⁻¹) .= parent(bⁿ)
    
    return nothing
end

function one_dimensional_simulation(grid, advection, label)

    auxiliary_fields = (; bⁿ⁻¹ = CenterField(grid))

    model = NonhydrostaticModel(; grid, auxiliary_fields, tracers = :b, timestepper = :QuasiAdamsBashforth2, advection) 

    set!(model.tracers.b, bᵢ)
    set!(model.velocities.u, 1.0)

    Δt = 0.1 * minimum_xspacing(grid)

    dissipation_computation = TracerVarianceDissipation(model; tracers = :b)
    simulation  = Simulation(model; Δt, stop_time = 10)
    simulation.callbacks[:compute_dissipation]    = Callback(dissipation_computation, IterationInterval(1))
    simulation.callbacks[:save_previous_buoyancy] = Callback(save_previous_buoyancy!, IterationInterval(1); callsite = Oceananigans.BeforeTimeStepCallsite())

    Px   = dissipation_computation.production.b.x
    bⁿ⁻¹ = simulation.model.auxiliary_fields.bⁿ⁻¹
    bⁿ   = model.tracers.b

    Σb²  = (bⁿ^2 - bⁿ⁻¹^2) / Δt # Not at the same time-step of Px unfortunately
    outputs  = (; Px, b = model.tracers.b, Σb²)
    filename = "dissipation_" * label * ".jld2"

    # Save the dissipation
    simulation.output_writers[:dissipation] = JLD2OutputWriter(model, outputs;
                                                               filename,
                                                               schedule = IterationInterval(10),
                                                               overwrite_existing = true)

    run!(simulation)
end

grid = RectilinearGrid(size = 100, halo = 6, x = (-1, 1), topology = (Periodic, Flat, Flat))

advections = [UpwindBiased(; order = 3), 
              WENO(; order = 5), 
              WENO(; order = 7)]

labels  = ["upwind3",
           "WENO5",
           "WENO7"]

# Run the simulations
for (advection, label) in zip(advections, labels)
    one_dimensional_simulation(grid, advection, label)
end

B_series = []
b_series = []
P_series = []

iter = Observable(1)

bn = []
Pn = []
Bn = []

for (i, label) in enumerate(labels)
    filename = "dissipation_" * label * ".jld2"

    push!(b_series, FieldTimeSeries(filename, "b"))
    push!(P_series, FieldTimeSeries(filename, "Px"))
    push!(B_series, FieldTimeSeries(filename, "Σb²"))

    push!(bn, @lift(interior(b_series[i][$iter], :, 1, 1)))
    push!(Pn, @lift(interior(P_series[i][$iter], :, 1, 1)))
    push!(Bn, @lift(interior(B_series[i][$iter], :, 1, 1)))
end

b0 = b_series[1][1]  
x  = xnodes(b0)

fig = Figure(size = (1200, 400))
ax  = Axis(fig[1, 1], xlabel = L"x", ylabel = L"tracer")
lines!(ax, x, interior(b0, :, 1, 1), color = :grey, linestyle = :dash, linewidth = 2, label = "initial condition")
lines!(ax, x, bn[1], label = labels[1], color = :red )
lines!(ax, x, bn[3], label = labels[2], color = :blue)
axislegend(ax, position = :rb)

ax  = Axis(fig[1, 2], xlabel = L"x", ylabel = L"variance dissipation")
lines!(ax, x, Pn[1], color = :red , label = labels[1])
lines!(ax, x, Pn[3], color = :blue, label = labels[2])
lines!(ax, x, Bn[1], color = :red , linestyle = :dash)
lines!(ax, x, Bn[3], color = :blue, linestyle = :dash)
ylims!(ax, (-1, 1))
axislegend(ax, position = :rb)

record(fig, "implicit_dissipation.mp4", 1:length(b_series[1]), framerate=8) do i
    @info "doing iter $i"
    iter[] = i
end

Nt = length(b_series[1])
time = (0:Nt-1) .* 0.1 .* minimum_xspacing(grid)

fig = Figure()
ax  = Axis(fig[1, 1], xlabel = L"time", ylabel = L"tracer variance budget")
Σb² = [sum(B_series[1][i]) for i in 1:Nt]
ΣP  = [sum(P_series[1][i]) for i in 1:Nt]
lines!(ax, time, Σb²,        label = "Σb²", color = :red)
lines!(ax, time, ΣP / 0.002, label = "ΣP",  color = :red, linestyle = :dash)
Σb² = [sum(B_series[2][i]) for i in 1:Nt]
ΣP  = [sum(P_series[2][i]) for i in 1:Nt]
lines!(ax, time, Σb²,        label = "Σb²", color = :blue)
lines!(ax, time, ΣP / 0.002, label = "ΣP",  color = :blue, linestyle = :dash)
Σb² = [sum(B_series[3][i]) for i in 1:Nt]
ΣP  = [sum(P_series[3][i]) for i in 1:Nt]
lines!(ax, time, Σb²,        label = "Σb²", color = :green)
lines!(ax, time, ΣP / 0.002, label = "ΣP",  color = :green, linestyle = :dash)