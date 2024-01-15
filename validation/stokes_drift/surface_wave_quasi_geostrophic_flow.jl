using Oceananigans
using Oceananigans.Units
using GLMakie

ϵ = 0.1
λ = 100 # meters
g = 9.81

const k = 2π / λ

c = sqrt(g / k)
const δ = 400kilometer
const cᵍ = c / 2
const Uˢ = ϵ^2 * c

@inline       A(ξ, η) = exp(- (ξ^2 + η^2) / 2δ^2)
@inline    ∂ξ_A(ξ, η) = - ξ / δ^2 * A(ξ, η)
@inline    ∂η_A(ξ, η) = - η / δ^2 * A(ξ, η)
@inline ∂η_∂ξ_A(ξ, η) = η * ξ / δ^4 * A(ξ, η)
@inline   ∂²ξ_A(ξ, η) = (ξ^2 / δ^2 - 1) * A(ξ, η) / δ^2

# Write the Stokes drift as
#
# uˢ(x, z, t) = A(x, t) * ûˢ(z)
#
# which implies

@inline    ûˢ(z)       = Uˢ * exp(2k * z)
@inline    uˢ(x, y, z, t) =           A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂z_uˢ(x, y, z, t) =     2k *  A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_uˢ(x, y, z, t) =        ∂η_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_uˢ(x, y, z, t) = - cᵍ * ∂ξ_A(x - cᵍ * t, y) * ûˢ(z)

# Note that if uˢ represents the solenoidal component of the Stokes drift,
# then
#
# ```math
# ∂z_wˢ = - ∂x_uˢ = - ∂ξ_A * ûˢ .
# ```
#
# We therefore find that
#
# ```math
# wˢ = - ∂ξ_A / 2k * ûˢ
# ```
#
# and

@inline ∂x_wˢ(x, y, z, t) = -  1 / 2k *   ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_wˢ(x, y, z, t) = -  1 / 2k * ∂η_∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_wˢ(x, y, z, t) = + cᵍ / 2k *   ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)

stokes_drift = StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂y_uˢ, ∂t_wˢ, ∂x_wˢ, ∂y_wˢ)

grid = RectilinearGrid(size = (64, 64, 16),
                       x = (-5δ, 15δ),
                       y = (-10δ, 10δ),
                       z = (-512, 0),
                       topology = (Periodic, Periodic, Bounded))

model = NonhydrostaticModel(; grid, stokes_drift,
                            tracers = :b,
                            coriolis = FPlane(f = 1e-4),
                            buoyancy = BuoyancyTracer(),
                            timestepper = :RungeKutta3)

# Set Lagrangian-mean flow equal to uˢ,
uᵢ(x, y, z) = uˢ(x, y, z, 0)

# And put in a stable stratification,
N² = 0 #1e-6
bᵢ(x, y, z) = N² * z
set!(model, u=uᵢ, b=bᵢ)

Δx = xspacings(grid, Center())
Δt = 0.2 * Δx / cᵍ
simulation = Simulation(model; Δt, stop_iteration = 200)

progress(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

filename = "surface_wave_quasi_geostrophic_induced_flow.jld2"
outputs = model.velocities
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs; filename,
                                                    schedule = IterationInterval(10),
                                                    overwrite_existing = true)

run!(simulation)

ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")

times = ut.times
Nt = length(times)
Nz = size(grid, 3)

n = Observable(1)

un = @lift interior(ut[$n], :, :, Nz)
wn = @lift interior(wt[$n], :, :, Nz)

xu, yu, zu = nodes(ut)
xw, yw, zw = nodes(wt)

fig = Figure(resolution=(800, 400))

axu = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", aspect=1)
axw = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", aspect=1)

heatmap!(axu, xu, yu, un)
heatmap!(axw, xw, yw, wn)

record(fig, "surface_wave_induced_flow.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

