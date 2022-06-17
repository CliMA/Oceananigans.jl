using Printf
using Oceananigans
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation
using Oceananigans.Advection: VelocityStencil, VorticityStencil
using Statistics: mean

arch = GPU()
Nh = 400
stencil = VelocityStencil

grid = RectilinearGrid(arch, size = (Nh, Nh), x = (0, 1), y = (0, 1), halo = (4, 4), topology = (Bounded, Bounded, Flat))

g = 1.0
f = 5.0
H  = 1.0
R = (g*H)^0.5 / f

model = ShallowWaterModel(grid = grid,
                        gravitational_acceleration = g,
                        coriolis = FPlane(f = f),
                        mass_advection = WENO(),
                        momentum_advection = WENO(vector_invariant = stencil()),
                        formulation = VectorInvariantFormulation())

# Model initialization
h₀ = 0.2
σ  = 0.07
d  = 1.4σ

gaussian(x, y, σ) = exp(-(x^2 + y^2)/(2*σ^2))
hᵢ(x, y, z) = H + h₀ * (gaussian(x - d - 0.5, y - 0.5, σ) + gaussian(x - 0.5 + d, y - 0.5, σ))

set!(model, h = hᵢ)
set!(model, u = - ∂y(model.solution.h) / f)
set!(model, v =   ∂x(model.solution.h) / f)
@info "Model initialized"

#####
##### Simulation setup
#####

g = model.gravitational_acceleration
gravity_wave_speed = sqrt(g * maximum(model.solution.h)) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = (grid.Lx / grid.Nx) / gravity_wave_speed

Δt = wave_propagation_time_scale * 0.1

simulation = Simulation(model, Δt = Δt, stop_time = 10)
start_time = [time_ns()]

function progress(sim)
    wall_time = (time_ns() - start_time[1]) * 1e-9

    u = sim.model.solution[1]

    @info @sprintf("Time: % 12s, iteration: %d, max(|u|): %.2e ms⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration, maximum(abs, u),
                    prettytime(wall_time))

    start_time[1] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

u, v, h = model.solution

Eᵦ = g * H * H

ζ = Field(∂x(v) - ∂y(u))
compute!(ζ)

KE = Field(h * (u^2 + v^2))
PE = Field(g * h^2)
compute!(KE)
compute!(PE)

save_interval = 0.1

simulation.output_writers[:surface_fields] = JLD2OutputWriter(model, (; u, v, h, ζ, KE, PE),
                                                            schedule = TimeInterval(save_interval),
                                                            filename = "vortex_merger_$(Nh)_$stencil",
                                                            overwrite_existing = true)

run!(simulation)

using JLD2

file = jldopen("vortex_merger_$(Nh)_$stencil.jld2")
iterations = parse.(Int, keys(file["timeseries/t"]))

ke2 = zeros(length(iterations))
pe2 = zeros(length(iterations))
z2  = zeros(length(iterations))
en  = zeros(length(iterations))
ett = zeros(length(iterations))
ztt = zeros(length(iterations))

for (idx, iter) in enumerate(iterations)
    ke2[idx] = mean(file["timeseries/KE/" * string(iter)][:, :, 1])
    pe2[idx] = mean(file["timeseries/PE/" * string(iter)][:, :, 1])
    en[idx]  = mean(ke2[idx] .+ pe2[idx])
    z2[idx]  = mean(file["timeseries/ζ/" * string(iter)][:, :, 1] .^ 2)
    ett[idx] = (en[1] - en[idx]) / (en[1] - Eᵦ)
    ztt[idx] = (z2[1] - z2[idx]) / z2[1]
end

jldsave("energy_vorticity_$(Nh)_$stencil.jld2", energy = ett, vorticity = ztt)

file = jldopen("vortex_merger_1600_VelocityStencil.jld2")
using CairoMakie

iter = Observable(0)
iterations = parse.(Int, keys(file["timeseries/t"]))

ζ′ = @lift(file["timeseries/ζ/" * string($iter)][:, 1:end-1, 1])
h′ = @lift(file["timeseries/h/" * string($iter)][:, 1:end-1, 1])
xζ, yζ, z = nodes((Face, Face, Center), grid)
xh, yh, z = nodes((Center, Center, Center), grid)

title = @lift(@sprintf("Vorticity in Shallow Water Model at time = %s", prettytime(file["timeseries/t/" * string($iter)])))
fig = CairoMakie.Figure(resolution = (1000, 600))

ax = CairoMakie.Axis(fig[1,1], xlabel = "longitude", ylabel = "latitude", title=title)
heatmap_plot = CairoMakie.heatmap!(ax, xζ, yζ, ζ′, colormap=Reverse(:balance), colorrange = (-9.0, 4.5))
CairoMakie.Colorbar(fig[1,2], heatmap_plot, width=25)

CairoMakie.record(fig, "vortex_merger.mp4", iterations[1:end], framerate=2) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    iter[] = i
end
