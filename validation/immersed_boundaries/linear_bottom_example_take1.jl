using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using CairoMakie
using Printf

# FJP: to do
# no slip on v at bottom
# db/dz = plus/minus N^2 
# nonisotropic diffusion

Nx, Nz = 100, 100
Lx, Lz = 1kilometers, 200meters

V∞ = -0.1 
N² = 1e-5 

α = 2e-2
bottom(x) = -Lz +  α*x

refinement = 1.8 # controls spacing near surface (higher means finer spaced)
stretching = 10  # controls rate of stretching at bottom 

h(k) = (Nz + 1 - k) / Nz
ζ(k) = 1 + (h(k) - 1) / refinement
Σ(k) = (1 - exp(-stretching * h(k))) / (1 - exp(-stretching))

z_faces(k) = - Lz * (ζ(k) * Σ(k) - 1) - Lz

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (0, Lx),
                                  z = z_faces,
                                  halo = (4, 4),
                                  topology = (Bounded, Flat, Bounded))


grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

x = xnodes(grid, Center())
bottom_boundary = interior(grid.immersed_boundary.bottom_height, :, 1, 1)
top_boundary = 0*x

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((0, grid.Lx/1e3), (-grid.Lz, 0)))

# FJP better to plot actual topography not just a line
band!(ax, x/1e3, bottom_boundary, top_boundary, color = :mediumblue)

fig

bottom_bc = ImmersedBoundaryCondition(bottom=ValueBoundaryCondition(0.0))
velocity_bcs = FieldBoundaryConditions(immersed=bottom_bc)

v_bcs = FieldBoundaryConditions(bottom = ValueBoundaryCondition(0.0));

model = HydrostaticFreeSurfaceModel(; grid, 
                                      coriolis = FPlane(f = 1e-4), 
                                      buoyancy = BuoyancyTracer(),
                                      free_surface = SplitExplicitFreeSurface(grid; cfl = 0.7),
                                      closure = ScalarDiffusivity(ν=1e-2, κ=1e-2),
                                      tracers = :b,
                                      #boundary_conditions = (v = velocity_bcs,),
                                      momentum_advection = UpwindBiasedFifthOrder(), 
                                      tracer_advection = UpwindBiasedFifthOrder())


vᵢ = V∞ 
bᵢ(x, z) = N² * z
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

filename = "linear_bathymetry_take1"
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, w, b);
                                                      filename,
                                                      schedule = TimeInterval(10minutes),
                                                      overwrite_existing = true)


run!(simulation)

saved_output_filename = filename * ".jld2"

u_t = FieldTimeSeries(saved_output_filename,  "u")
v_t = FieldTimeSeries(saved_output_filename,  "v")
w_t = FieldTimeSeries(saved_output_filename,  "w")
b_t = FieldTimeSeries(saved_output_filename,  "b")

umax = maximum(abs, u_t[end])
vmax = maximum(abs, v_t[end])
wmax = maximum(abs, w_t[end])
bmax = maximum(abs, b_t[end])

times = u_t.times

for φ_t in (u_t, v_t, w_t, b_t), n in 1:length(times)
    mask_immersed_field!(φ_t[n], NaN)
end

xu,  yu,  zu  = nodes(u_t[1]) ./ 1e3
xv,  yv,  zv  = nodes(v_t[1]) ./ 1e3
xw,  yw,  zw  = nodes(w_t[1])  ./ 1e3
xb,  yb,  zb  = nodes(b_t[1])  ./ 1e3

n = Observable(1)

title = @lift @sprintf("t = %1.2f hours",
                       round(times[$n] / hours, digits=2))

uₙ = @lift u_t[1:Nx, 1, 1:Nz, $n]
vₙ = @lift v_t[1:Nx, 1, 1:Nz, $n]
wₙ = @lift w_t[1:Nx, 1, 1:Nz, $n]
bₙ = @lift b_t[1:Nx, 1, 1:Nz, $n]

axis_kwargs = (xlabel = "x [km]",
               ylabel = "z [km]",
               limits = ((0, grid.Lx/1e3), (-grid.Lz/1e3, -grid.Lz/4e3)),
               titlesize = 20)

fig = Figure(size = (700, 900))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_v = Axis(fig[2, 1]; title = "v", axis_kwargs...)
hm_v = heatmap!(ax_v, xv, zv, vₙ; colorrange = (-0.103, -0.099), colormap = :balance)
Colorbar(fig[2, 2], hm_v, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w", axis_kwargs...)
hm_w = heatmap!(ax_w, xw, zw, wₙ; colorrange = (-1e-4, 2e-4), colormap = :balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

ax_b = Axis(fig[4, 1]; title = "b", axis_kwargs...)
hm_b = heatmap!(ax_b, xb, zb, bₙ; colorrange = (-0.002, 0), colormap = :balance)
Colorbar(fig[4, 2], hm_b, label = "m s⁻¹")

fig

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=16) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end

fig = Figure(size = (700, 700))

ax_v = Axis(fig[1, 1]; title = "v (take 1)", axis_kwargs...)
hm_v = heatmap!(ax_v, xv, zv, vₙ; colormap = :balance)
cn_b = contour!(ax_v, xb, zb, bₙ, levels=30, color="black")
Colorbar(fig[1,2], hm_v, label = "m s⁻¹")

save("v_b_final_take1.png", fig)