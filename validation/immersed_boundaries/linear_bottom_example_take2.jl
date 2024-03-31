using Oceananigans
using Oceananigans.Units
using NCDatasets
using CairoMakie
using Printf

Nx, Nz = 100, 100
Lx, Lz = 1kilometers, 200meters

V∞ = -0.1 
N² = 1e-5 

α = 2e-2                      # tilted bottom
θ = atan(α)/pi*180
ĝ = [sind(θ), 0, cosd(θ)]

refinement = 1.8              # stretched grid
stretching = 10  

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1) - Lz

grid = RectilinearGrid(size = (Nx, Nz),
                       x = (0, Lx),
                       z = z_faces,
                       topology = (Bounded, Flat, Bounded))

lines(zspacings(grid, Center()), znodes(grid, Center()),
      axis = (ylabel = "Depth (m)",
              xlabel = "Vertical spacing (m)"))

scatter!(zspacings(grid, Center()), znodes(grid, Center()))

current_figure() #hide

#z₀ = 0.1 
#κ  = 0.4 

#z₁ = first(znodes(grid, Center())) # Closest grid center to the bottom
#cᴰ = (κ / log(z₁ / z₀))^2 # Drag coefficient

#@inline drag_u(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * u
#@inline drag_v(x, t, u, v, p) = - p.cᴰ * √(u^2 + (v + p.V∞)^2) * (v + p.V∞)

#drag_bc_u = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))
#drag_bc_v = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=(; cᴰ, V∞))

#u_bcs = FieldBoundaryConditions(bottom = drag_bc_u)
#v_bcs = FieldBoundaryConditions(bottom = drag_bc_v)

v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0));

model = NonhydrostaticModel(; grid, 
                            coriolis = ConstantCartesianCoriolis(f = 1e-4, rotation_axis = ĝ),
                            buoyancy = Buoyancy(model = BuoyancyTracer(), gravity_unit_vector = -ĝ),
                            closure = ScalarDiffusivity(ν=1e-2, κ=1e-2),
                            tracers = :b,
                            #boundary_conditions = (v = v_bcs,),
                            timestepper = :RungeKutta3,
                            advection = UpwindBiasedFifthOrder())

vᵢ = V∞
bᵢ(x, z) = N² * (x * ĝ[1] + z * ĝ[3])
set!(model, v = vᵢ, b=bᵢ)

simulation = Simulation(model, Δt = 0.5 * minimum_zspacing(grid) / V∞, stop_time = 12hours)

wizard = TimeStepWizard(max_change=1.1, cfl=0.1, min_Δt = 0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(4))

start_time = time_ns() # so we can print the total elapsed wall time

progress_message(sim) =
    @printf("Iteration: %04d, time: %s, Δt: %s, max|w|: %.1e m s⁻¹, wall time: %s\n",
            iteration(sim), prettytime(time(sim)),
            prettytime(sim.Δt), maximum(abs, sim.model.velocities.w),
            prettytime((time_ns() - start_time) * 1e-9))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(20))

u, v, w = model.velocities
b = model.tracers.b

simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; u, v, w, b);
                                                        filename = joinpath(@__DIR__, "tilted_bottom_boundary_layer.nc"),
                                                        schedule = TimeInterval(10minutes),
                                                        overwrite_existing = true)


run!(simulation)

xv, yv, zv = nodes(v)
xw, yw, zw = nodes(w)
xb, yb, zb = nodes(b)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

#umax = maximum(abs, ds["u"])
#vmax = maximum(abs, ds["v"])
#wmax = maximum(abs, ds["w"])
#bmax = maximum(abs, ds["b"])

fig = Figure(size = (800, 600))

axis_kwargs = (xlabel = "Across-slope distance (m)",
               ylabel = "Slope-normal\ndistance (m)",
               limits = ((0, Lx), (-Lz, -Lz/4)),
               )

ax_v = Axis(fig[2, 1]; title = "v", axis_kwargs...)
ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
ax_b = Axis(fig[4, 1]; title = "b", axis_kwargs...)

n = Observable(1)

vₙ = @lift ds["v"][:, 1, :, $n]
hm_v = heatmap!(ax_v, xv, zv, vₙ, colorrange = (-0.103, -0.099), colormap = :balance)
Colorbar(fig[2, 2], hm_v; label = "m s⁻¹")

wₙ = @lift ds["w"][:, 1, :, $n]
hm_w = heatmap!(ax_w, xw, zw, wₙ, colorrange = (-1e-4, 2e-4), colormap = :balance)
Colorbar(fig[3, 2], hm_w; label = "m s⁻¹")

bₙ = @lift ds["b"][:, 1, :, $n]
hm_b = heatmap!(ax_b, xb, zb, bₙ, colorrange = (-0.002, 0), colormap = :balance)
Colorbar(fig[4, 2], hm_b; label = "m s⁻¹")

times = collect(ds["time"])
title = @lift "t = " * string(prettytime(times[$n]))
fig[1, :] = Label(fig, title, fontsize=20, tellwidth=false)

current_figure() #hide
fig

frames = 1:length(times)

record(fig, "linear_bathymetry_take2.mp4", frames, framerate=12) do i
    n[] = i
end
nothing #hide

close(ds)

fig = Figure(size = (700, 700))

ax_v = Axis(fig[1, 1]; title = "v (take 2)", axis_kwargs...)
hm_v = heatmap!(ax_v, xv, zv, vₙ; colormap = :balance)
cn_b = contour!(ax_v, xb, zb, bₙ, levels=30, color="black")
Colorbar(fig[1,2], hm_v, label = "m s⁻¹")

save("v_b_final_take2.png", fig)