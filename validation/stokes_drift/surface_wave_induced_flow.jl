using Oceananigans
using Oceananigans.Units
using GLMakie

ϵ = 0.1
λ = 60 # meters
g = 9.81

const k = 2π / λ

c = sqrt(g / k)
const δ = 1kilometer
const cᵍ = c / 2
const Uˢ = ϵ^2 * c

@inline A(ξ) = exp(- ξ^2 / 2δ^2)
@inline A′(ξ) = - ξ / δ^2 * A(ξ)
@inline A′′(ξ) = (ξ^2 / δ^2 - 1) * A(ξ) / δ^2

# Write the Stokes drift as
#
# uˢ(x, z, t) = A(x, t) * ûˢ(z)
#
# which implies

@inline    ûˢ(z)       = Uˢ * exp(2k * z)
@inline    uˢ(x, z, t) =         A(x - cᵍ * t) * ûˢ(z)
@inline ∂z_uˢ(x, z, t) =   2k *  A(x - cᵍ * t) * ûˢ(z)
@inline ∂t_uˢ(x, z, t) = - cᵍ * A′(x - cᵍ * t) * ûˢ(z)

# Note that if uˢ represents the solenoidal component of the Stokes drift,
# then
#
# ```math
# ∂z_wˢ = - ∂x_uˢ = - A′ * ûˢ .
# ```
#
# We therefore find that
#
# ```math
# wˢ = - A′ / 2k * ûˢ
# ```
#
# and

@inline ∂x_wˢ(x, z, t) = -  1 / 2k * A′′(x - cᵍ * t) * ûˢ(z)
@inline ∂t_wˢ(x, z, t) = + cᵍ / 2k * A′′(x - cᵍ * t) * ûˢ(z)

stokes_drift = StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂t_wˢ, ∂x_wˢ)

grid = RectilinearGrid(size = (256, 64),
                       x = (-5kilometers, 15kilometers),
                       z = (-512, 0),
                       topology = (Periodic, Flat, Bounded))

model = NonhydrostaticModel(; grid, stokes_drift,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            timestepper = :RungeKutta3)

# Set Lagrangian-mean flow equal to uˢ,
uᵢ(x, z) = uˢ(x, z, 0)

# And put in a stable stratification,
N² = 0
bᵢ(x, z) = N² * z
set!(model, u=uᵢ, b=bᵢ)

Δx = xspacings(grid, Center())
Δt = 0.2 * Δx / cᵍ
simulation = Simulation(model; Δt, stop_iteration = 600)

progress(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

filename = "surface_wave_induced_flow.jld2"
outputs = model.velocities
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs; filename,
                                                    schedule = IterationInterval(10),
                                                    overwrite_existing = true)

run!(simulation)

ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")

times = ut.times
Nt = length(times)

n = Observable(1)

un = @lift interior(ut[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)

xu, yu, zu = nodes(ut)
xw, yw, zw = nodes(wt)

fig = Figure(resolution=(800, 300))

axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")

heatmap!(axu, xu, zu, un)
heatmap!(axw, xw, zw, wn)

record(fig, "surface_wave_induced_flow.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

