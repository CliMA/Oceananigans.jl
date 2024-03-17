using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures:
    RiBasedVerticalDiffusivity,
    CATKEVerticalDiffusivity,
    TKEDissipationVerticalDiffusivity,
    ConvectiveAdjustmentVerticalDiffusivity,
    ExplicitTimeDiscretization

# Parameters
Δz = 4          # Vertical resolution
Lz = 256        # Extent of vertical domain
Nz = Int(Lz/Δz) # Vertical resolution
f₀ = 1e-4       # Coriolis parameter (s⁻¹)
N² = 1e-6       # Buoyancy gradient (s⁻²)
Jᵇ = +1e-8      # Surface buoyancy flux (m² s⁻³)
τˣ = -2e-4      # Surface kinematic momentum flux (m s⁻¹)
stop_time = 2days

# Set up simulation
grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=f₀)
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τˣ))
closure = TKEDissipationVerticalDiffusivity()

tracers = (:b, :e, :ϵ)
closure_initial_conditions = (; e=1e-6, ϵ=1e-3)

model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis, tracers,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs))
                                    
bᵢ(z) = N² * z
set!(model; b=bᵢ, closure_initial_conditions...)

Δt = 10.0 #1minutes
#simulation = Simulation(model; Δt, stop_time)
simulation = Simulation(model; Δt, stop_iteration=100) #stop_time)
pop!(simulation.callbacks, :nan_checker)

closurename = string(nameof(typeof(closure)))

diffusivities = (κu = model.diffusivity_fields.κu,
                 κc = model.diffusivity_fields.κc)

outputs = merge(model.velocities, model.tracers, diffusivities)

simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      #schedule = TimeInterval(20minutes),
                                                      schedule = IterationInterval(10),
                                                      filename = "windy_convection_k_epsilon.jld2",
                                                      overwrite_existing = true)

function progress(sim)
    msg = string("Iter: ", iteration(sim), " t: ", prettytime(sim))

    b, e, ϵ = model.tracers
    msg *= @sprintf(", max(b): %.2e, extrema(e): (%.2e, %.2e), extrema(ϵ): (%.2e, %.2e)",
                    maximum(b),
                    minimum(e), maximum(e),
                    minimum(ϵ), maximum(ϵ))

    @info msg

    return nothing
end
                             
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

@info "Running a simulation of $model..."

run!(simulation)

#####
##### Visualize
#####

filepath = "windy_convection_k_epsilon.jld2"

bt = FieldTimeSeries(filepath, "b")
ut = FieldTimeSeries(filepath, "u")
vt = FieldTimeSeries(filepath, "v")
et = FieldTimeSeries(filepath, "e")
ϵt = FieldTimeSeries(filepath, "ϵ")
κct = FieldTimeSeries(filepath, "κc")
κut = FieldTimeSeries(filepath, "κu")

zc = znodes(bt)
zf = znodes(κct)
Nt = length(bt.times)

fig = Figure(size=(1800, 600))

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(bt.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(bt.times[$n])
TKE_label = @lift "Turbulent kinetic energy t = " * prettytime(bt.times[$n])
diffusivities_label = @lift "Eddy diffusivities at t = " * prettytime(bt.times[$n])

axb = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z (m)")
axu = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z (m)")
axe = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z (m)")
axκ = Axis(fig[1, 4], xlabel=diffusivities_label, ylabel="z (m)")

xlims!(axb, -grid.Lz * N², 0)
xlims!(axu, -0.1, 0.1)
xlims!(axe, -1e-4, 2e-4)
xlims!(axκ, -1e-1, 5e-1)

colors = [:black, :blue, :red, :orange]

bn  = @lift interior(bt[$n], 1, 1, :)
un  = @lift interior(ut[$n], 1, 1, :)
vn  = @lift interior(vt[$n], 1, 1, :)
en  = @lift interior(et[$n], 1, 1, :)
κcn = @lift interior(κct[$n], 1, 1, :)
κun = @lift interior(κut[$n], 1, 1, :)

closurename = string(nameof(typeof(closure)))

lines!(axb, bn,  zc, label=closurename)
lines!(axu, un,  zc, label="u")
lines!(axu, vn,  zc, label="v", linestyle=:dash)
lines!(axe, en,  zc, label="e")
lines!(axκ, κcn, zf, label="κc")
lines!(axκ, κun, zf, label="κu", linestyle=:dash)

axislegend(axb, position=:lb)
axislegend(axu, position=:rb)
axislegend(axe, position=:rb)
axislegend(axκ, position=:rb)

display(fig)

# record(fig, "windy_convection.mp4", 1:Nt, framerate=24) do nn
#     n[] = nn
# end

