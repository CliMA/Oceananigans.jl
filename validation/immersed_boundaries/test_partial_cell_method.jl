using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: mask_immersed_field!, PartialCellBottom
using Oceananigans.ImmersedBoundaries: Δzᶜᶜᶠ, Δzᶜᶜᶜ, immersed_cell, bottom_cell

using CairoMakie
using Printf

#FJP halos are not filled!!!!


# FJP: to do
# no slip on v at bottom
# db/dz = plus/minus N^2 
# nonisotropic diffusion

Nx, Nz = 5, 5
Lx, Lz = 1kilometers, 200meters

α = 0.1 #2e-2
bottom(x) = -Lz +  α*x

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (0, Lx),
                                  z = (-Lz, 0),
                                  halo = (4, 4),
                                  topology = (Bounded, Flat, Bounded))


grid_gfb = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))
grid_pcb = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom, minimum_fractional_cell_height=0.1))

xᶜ,xᶠ = xnodes(underlying_grid, Center()), xnodes(underlying_grid, Face())
zᶜ,zᶠ = znodes(underlying_grid, Center()), znodes(underlying_grid, Face())

# bottom cell agrees with what we find in k_gfb
 z_gfb, z_pcb = zeros(Float64, Nx), zeros(Float64, Nx)
 k_gfb, k_pcb = zeros(Int64, Nx), zeros(Int64, Nx)
 for (i, x) in enumerate(xᶜ)
    k = findlast(zᶠ .<= grid_pcb.immersed_boundary.bottom_height.data[i, 1, 1])
    k_gfb[i] = k
    z_gfb[i] = zᶠ[k]  
    z_pcb[i] = zᶠ[k+1] - Δzᶜᶜᶜ(i, 1, k, grid_pcb)      
 end

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1],
          title="Bathymetry",
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((0, Lx/1e3), (-Lz*1.1, 0)))

# Plot vertical grid lines
for (i, x) in enumerate(xᶠ)
    lines!(ax, [x/1e3, x/1e3], [-Lz, 0], linewidth=1, color=:grey)
end

# Plot horizontal grid lines
for (k, z) in enumerate(zᶠ)
    lines!(ax, [0, Lx/1e3], [z, z], linewidth=1, color=:grey)
end

# Plot bottom function 
bottom_faces = bottom.(xᶠ)
lines!(ax, xᶠ/1e3, bottom_faces, linewidth=4, color=:red, label="-200 +  0.2*x")

# Plot GridFittedBottom
stairs!(ax, vcat(xᶠ[1], xᶜ, xᶠ[end])/1e3, vcat(z_gfb[1], z_gfb, z_gfb[end]), 
            linewidth = 4, step=:center, color=:blue, label="GridFittedBottom")
scatter!(ax, xᶜ/1e3, z_gfb, markersize = 10, color=:blue)

# Plot PartialCellBottom
stairs!(ax, vcat(xᶠ[1], xᶜ, xᶠ[end])/1e3, vcat(z_pcb[1], z_pcb, z_pcb[end]), 
            linewidth = 4, step=:center, color=:green, label="PartialCellBottom")
scatter!(ax, xᶜ/1e3, z_pcb, markersize = 10, color=:green)

axislegend(ax; position = :lt)

save("CompareBottoms.png", fig)


## Plot the dual grid

zdual_gfb, zdual_pcb = zeros(Float64, Nx), zeros(Float64, Nx)
kdual_gfb, kdual_pcb = zeros(Int64, Nx), zeros(Int64, Nx)
for (i, x) in enumerate(xᶜ)
    k = findlast(zᶜ .<= grid_pcb.immersed_boundary.bottom_height.data[i, 1, 1])
    if sizeof(k) == 0
        kdual_gfb[i] = 1
        zdual_gfb[i] = zᶜ[1]  
        zdual_pcb[i] = zᶜ[1]      
    else
        kdual_gfb[i] = k
        zdual_gfb[i] = zᶜ[k]  
        zdual_pcb[i] = zᶜ[k+1] - Δzᶜᶜᶠ(i, 1, k+1, grid_pcb)      
    end
end

fig = Figure(size = (700, 400))
ax = Axis(fig[1, 1],
          title="Bathymetry (dual grid)",
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((0, Lx/1e3), (-Lz*1.1, 0)))

# Plot vertical grid lines
for (i, x) in enumerate(xᶜ)
    lines!(ax, [x/1e3, x/1e3], [-Lz, 0], linewidth=1, color=:grey)
end

# Plot horizontal grid lines
for (k, z) in enumerate(zᶜ)
    lines!(ax, [0, Lx/1e3], [z, z], linewidth=1, color=:grey)
end

# Plot bottom function 
bottom_faces = bottom.(xᶠ)
lines!(ax, xᶠ/1e3, bottom_faces, linewidth=4, color=:red, label="-200 +  0.2*x")

# Plot GridFittedBottom
stairs!(ax, vcat(xᶠ[1], xᶜ, xᶠ[end])/1e3, vcat(zdual_gfb[1], zdual_gfb, zdual_gfb[end]),
            linewidth = 4, step=:center, color=:blue, label="GridFittedBottom")
scatter!(ax, xᶜ/1e3, zdual_gfb, markersize = 10, color=:blue)

# Plot PartialCellBottom
#stairs!(ax, vcat(xᶠ[1], xᶜ, xᶠ[end])/1e3, vcat(z_pcb[1], zdual_pcb, zdual_pcb[end]), 
#            linewidth = 4, step=:center, color=:green, label="PartialCellBottom")
#scatter!(ax, xᶜ/1e3, zdual_pcb, markersize = 10, color=:green)

axislegend(ax; position = :lt)

save("CompareBottoms_dual.png", fig)

#fig
#z_mat = repeat(zᶠ, 1, Nx+1)
#h_mat = repeat(transpose(grid_pcb.immersed_boundary.bottom_height.data[1:6, 1, 1]), Nz+1, 1)

#for (i, x) in enumerate(xᶜ), (k, z) in enumerate(zᶠ)
#    println(i, x, k, z)
#end

#=
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

=#