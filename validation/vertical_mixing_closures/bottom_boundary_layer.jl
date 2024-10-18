using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity, MixingLength

# Parameters
Δz = 4          # Vertical resolution
Lz = 256        # Extent of vertical domain
Nz = Int(Lz/Δz) # Vertical resolution
f₀ = 0.0        # Coriolis parameter (s⁻¹)
N² = 1e-6       # Buoyancy gradient (s⁻²)
ℓ₀ = 1e-4       # Roughness length
ϰ  = 0.4        # "Von Karman constant"
u₀ = 1.0        # Initial bottom velocity
stop_time = 1days

mixing_length = MixingLength(Cᵇ=0.1)
closure= CATKEVerticalDiffusivity(; mixing_length)

# Set up simulation

grid = RectilinearGrid(size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=f₀)

# Fluxes from similarity theory...
Cᴰ = (ϰ / log(Δz/2ℓ₀))^2
@inline τˣ(t, u, v, Cᴰ) = - Cᴰ * u * sqrt(u^2 + v^2)
@inline τʸ(t, u, v, Cᴰ) = - Cᴰ * v * sqrt(u^2 + v^2)

u_bottom_bc = FluxBoundaryCondition(τˣ, field_dependencies=(:u, :v), parameters=Cᴰ)
v_bottom_bc = FluxBoundaryCondition(τʸ, field_dependencies=(:u, :v), parameters=Cᴰ)

u_bcs = FieldBoundaryConditions(bottom = u_bottom_bc)
v_bcs = FieldBoundaryConditions(bottom = v_bottom_bc)

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; u=u_bcs, v=v_bcs))
                                    
bᵢ(z) = N² * z
set!(model, b=bᵢ, u=u₀, e=1e-6)

simulation = Simulation(model; Δt=2minutes, stop_time)

diffusivities = (κᵘ = model.diffusivity_fields.κᵘ,
                 κᶜ = model.diffusivity_fields.κᶜ)

outputs = merge(model.velocities, model.tracers, diffusivities)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = TimeInterval(20minutes),
                                                      filename = "bottom_boundary_layer.jld2",
                                                      overwrite_existing = true)

function progress(sim)
    msg = @sprintf("Iter: %d, time: %s, max(u): %.2f",
                   iteration(sim), prettytime(sim), maximum(model.velocities.u))

    @info msg
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

@info "Running a simulation of $model..."

run!(simulation)

filename = "bottom_boundary_layer.jld2"
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")
bt = FieldTimeSeries(filename, "b")
et = FieldTimeSeries(filename, "e")
κct = FieldTimeSeries(filename, "κᶜ")

Nt = length(ut)
zc = znodes(ut)
zf = znodes(κct)

fig = Figure(size=(1600, 400))
axu = Axis(fig[1, 1], ylabel="z (m)", xlabel="u, v (m s⁻¹)")
axb = Axis(fig[1, 2], ylabel="z (m)", xlabel="b (m s⁻²)")
axe = Axis(fig[1, 3], ylabel="z (m)", xlabel="e (m² s⁻²)")
axκ = Axis(fig[1, 4], ylabel="z (m)", xlabel="κ (m² s⁻¹)")

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

un = @lift interior(ut[$n], 1, 1, :)
vn = @lift interior(vt[$n], 1, 1, :)
bn = @lift interior(bt[$n], 1, 1, :)
en = @lift interior(et[$n], 1, 1, :)
κcn = @lift interior(κct[$n], 1, 1, :)

lines!(axu, un, zc, label="u")
lines!(axu, vn, zc, linestyle=:dash, label="v")
lines!(axb, bn, zc)
lines!(axe, en, zc)
lines!(axκ, κcn, zf)

xlims!(axe, -1e-5, 1e-3)
xlims!(axκ, -1e-5, 1e-1)

axislegend(axu)

display(fig)

record(fig, "bottom_boundary_layer.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end


