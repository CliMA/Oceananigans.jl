using GLMakie
using Oceananigans
using Oceananigans.Units
using Printf

using Oceananigans.TurbulenceClosures:
    RiBasedVerticalDiffusivity,
    CATKEVerticalDiffusivity,
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

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=0.1, convective_νz=0.01)
catke = CATKEVerticalDiffusivity()
ri_based = RiBasedVerticalDiffusivity()

# Set up simulation

grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
coriolis = FPlane(f=f₀)
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(τˣ))
#closures_to_run = [catke, ri_based, convective_adjustment]
closures_to_run = [catke] #, ri_based] #, convective_adjustment]

for closure in closures_to_run

    model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; b=b_bcs, u=u_bcs))
                                        
    bᵢ(z) = N² * z
    set!(model, b=bᵢ, e=1e-6)

    simulation = Simulation(model; Δt=2minutes, stop_iteration=1) #stop_time)

    closurename = string(nameof(typeof(closure)))

    diffusivities = (κᵘ = model.diffusivity_fields.κᵘ,
                     κᶜ = model.diffusivity_fields.κᶜ)

    outputs = merge(model.velocities, model.tracers, diffusivities)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          schedule = IterationInterval(1),
                                                          filename = "new_windy_convection_" * closurename,
                                                          overwrite_existing = true)

    progress(sim) = @info string("Iter: ", iteration(sim), " t: ", prettytime(sim),
                                 ", max(b): ", maximum(model.tracers.b))
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    @info "Running a simulation of $model..."

    run!(simulation)
end

include("compare_catke_results.jl")

#=
#####
##### Visualize
#####

b_ts = []
u_ts = []
v_ts = []
e_ts = []
κᶜ_ts = []
κᵘ_ts = []

for closure in closures_to_run
    closurename = string(nameof(typeof(closure)))
    filepath = "windy_convection_" * closurename * ".jld2"

    push!(b_ts, FieldTimeSeries(filepath, "b"))
    push!(u_ts, FieldTimeSeries(filepath, "u"))
    push!(v_ts, FieldTimeSeries(filepath, "v"))
    push!(e_ts, FieldTimeSeries(filepath, "e"))
    push!(κᶜ_ts, FieldTimeSeries(filepath, "κᶜ"))
    push!(κᵘ_ts, FieldTimeSeries(filepath, "κᵘ"))
end

b1 = first(b_ts)
e1 = first(e_ts)
κ1 = first(κᶜ_ts)
@show maximum(e1)

zc = znodes(b1)
zf = znodes(κ1)
Nt = length(b1.times)

fig = Figure(size=(1800, 600))

slider = Slider(fig[2, 1:4], range=1:Nt, startvalue=1)
n = slider.value

buoyancy_label = @lift "Buoyancy at t = " * prettytime(b1.times[$n])
velocities_label = @lift "Velocities at t = " * prettytime(b1.times[$n])
TKE_label = @lift "Turbulent kinetic energy t = " * prettytime(b1.times[$n])
diffusivities_label = @lift "Eddy diffusivities at t = " * prettytime(b1.times[$n])

axb = Axis(fig[1, 1], xlabel=buoyancy_label, ylabel="z (m)")
axu = Axis(fig[1, 2], xlabel=velocities_label, ylabel="z (m)")
axe = Axis(fig[1, 3], xlabel=TKE_label, ylabel="z (m)")
axκ = Axis(fig[1, 4], xlabel=diffusivities_label, ylabel="z (m)")

xlims!(axb, -grid.Lz * N², 0)
xlims!(axu, -0.1, 0.1)
xlims!(axe, -1e-4, 2e-4)
xlims!(axκ, -1e-1, 5e-1)

colors = [:black, :blue, :red, :orange]

for (i, closure) in enumerate(closures_to_run)
    bn  = @lift interior(b_ts[i][$n], 1, 1, :)
    un  = @lift interior(u_ts[i][$n], 1, 1, :)
    vn  = @lift interior(v_ts[i][$n], 1, 1, :)
    en  = @lift interior(e_ts[i][$n], 1, 1, :)
    κᶜn = @lift interior(κᶜ_ts[i][$n], 1, 1, :)
    κᵘn = @lift interior(κᵘ_ts[i][$n], 1, 1, :)
    
    closurename = string(nameof(typeof(closure)))

    lines!(axb, bn,  zc, label=closurename, color=colors[i])
    lines!(axu, un,  zc, label="u, " * closurename, color=colors[i])
    lines!(axu, vn,  zc, label="v, " * closurename, linestyle=:dash, color=colors[i])
    lines!(axe, en,  zc, label="e, " * closurename, color=colors[i])
    lines!(axκ, κᶜn, zf, label="κᶜ, " * closurename, color=colors[i])
    lines!(axκ, κᵘn, zf, label="κᵘ, " * closurename, linestyle=:dash, color=colors[i])
end

axislegend(axb, position=:lb)
axislegend(axu, position=:rb)
axislegend(axe, position=:rb)
axislegend(axκ, position=:rb)

display(fig)

# record(fig, "windy_convection.mp4", 1:Nt, framerate=24) do nn
#     n[] = nn
# end
=#
