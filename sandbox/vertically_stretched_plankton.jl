using Printf
using Oceananigans
using Oceananigans.Fields
using Oceananigans.OutputWriters
using Oceananigans.Advection
using Oceananigans.Utils

Nx = Nz = 64
Lx = Lz = 64
S = 2  # stretching factor
zF = -Lz .* (1 .+ [tanh(S * (2 * (k - 1) / 2Nz - 1)) / tanh(S) for k in 1:2Nz][1:Nz+1]) |> reverse
zF[end] = 0

grid = VerticallyStretchedCartesianGrid(size=(Nx, 1, Nz), x=(0, Lx), y=(0, Lx), zF=zF, halo=(3, 3, 3))

buoyancy_flux(x, y, t, p) = p.initial_buoyancy_flux * exp(-t^4 / (24 * p.shut_off_time^4))

buoyancy_flux_parameters = (initial_buoyancy_flux = 1e-8, shut_off_time = 6hours)
buoyancy_flux_bc = BoundaryCondition(Flux, buoyancy_flux, parameters = buoyancy_flux_parameters)

N² = 1e-4 # s⁻²
buoyancy_gradient_bc = BoundaryCondition(Gradient, N²)

buoyancy_bcs = TracerBoundaryConditions(grid, top = buoyancy_flux_bc, bottom = buoyancy_gradient_bc)

growing_and_grazing(x, y, z, t, P, p) = (p.μ₀ * exp(z / p.λ) - p.m) * P
plankton_dynamics_parameters = (μ₀ = 1/day,   # surface growth rate
                                 λ = 5,       # sunlight attenuation length scale (m)
                                 m = 0.1/day) # mortality rate due to virus and zooplankton grazing

plankton_dynamics = Forcing(growing_and_grazing, field_dependencies = :P,
                            parameters = plankton_dynamics_parameters)

model = IncompressibleModel(
                   grid = grid,
              advection = UpwindBiasedFifthOrder(),
            timestepper = :RungeKutta3,
                closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
               coriolis = FPlane(f=1e-4),
                tracers = (:b, :P), # P for Plankton
               buoyancy = BuoyancyTracer(),
                forcing = (P=plankton_dynamics,),
    boundary_conditions = (b=buoyancy_bcs,)
)

mixed_layer_depth = 32 # m
stratification(z) = z < -mixed_layer_depth ? N² * z : - N² * mixed_layer_depth
noise(z) = 1e-4 * N² * grid.Lz * randn() * exp(z / 4)

initial_buoyancy(x, y, z) = stratification(z) + noise(z)

set!(model, b=initial_buoyancy, P=1)

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

U_max = 0.05  # estimate
Δt = minimum(grid.ΔzF[1:Nz]) / U_max
simulation = Simulation(model, Δt=Δt, stop_time=24hours,
                        iteration_interval=10, progress=progress)

averaged_plankton = AveragedField(model.tracers.P, dims=(1, 2))

outputs = (w = model.velocities.w,
           plankton = model.tracers.P,
           averaged_plankton = averaged_plankton)

simulation.output_writers[:simple_output] =
    NetCDFOutputWriter(model, outputs, schedule = TimeInterval(10minutes),
                       filepath = "convecting_plankton.nc", mode = "c")

run!(simulation)

#####
##### Plotting
#####

# using GLMakie
# using GeometryBasics

# ds = NCDataset("convecting_plankton.nc")

# frame = Node(1)
# plot_title = @lift @sprintf("Vertically stretched plankton: t = %s", prettytime(ds["time"][$frame]))
# w = @lift ds["w"][:, 1, :, $frame]
# plankton = @lift ds["plankton"][:, 1, :, $frame]

# fig = Figure(resolution=(1920, 1080))

# ax_w = Axis(fig, xlabel="x (m)", ylabel="z (m)")
# hm_w = CairoMakie.heatmap!(ax_w, ds["xC"], ds["zF"], w, colormap=:balance, colorrange=(0.015, 0.015))
# cb_w = fig[:, end] = Colorbar(fig, hm_w, label="Vertical velocity w (m/s)", width=30, height=Relative(2/3))

# ax_p = Axis(fig, xlabel="x (m)", ylabel="z (m)")
# hm_p = CairoMakie.heatmap!(ax_p, ds["xC"], ds["zC"], plankton, colormap=:matter, colorrange=(0.95, 1.15))
# cb_p = fig[:, end] = Colorbar(fig, hm_w, label="Vertical velocity w (m/s)", width=30, height=Relative(2/3))

# # xlims!(ax, [0, 4π])
# # ylims!(ax, [(r-1)*π, r*π])

# supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)
# trim!(fig.layout)

# record(fig, "vertically_stretched_plankton.mp4", 1:length(ds[1]["time"])-1, framerate=30) do n
#     @info "Animating vertically stretched plankton frame $n/$(length(ds[1]["time"]))..."
#     frame[] = n
# end

# close(ds)

# # Code credit: https://github.com/JuliaPlots/Makie.jl/issues/675#issuecomment-706284015
# function stretched_heatmap(xs, ys, zs)
#     # Needs to be a mesh to not get centered?
#     m = Rect2D(Point2f0(0.0), Vec2f0(1)) |> normal_mesh

#     # midpoints
#     midxs = 0.5(xs[1:end-1] .+ xs[2:end])
#     midys = 0.5(ys[1:end-1] .+ ys[2:end])

#     # Padding for outer rectangles
#     midxs = [2xs[1] - midxs[1]; midxs; 2xs[end] - midxs[end]]
#     midys = [2ys[1] - midys[1]; midys; 2ys[end] - midys[end]]

#     # rectangle position & size
#     pos = [Point2f0(x, y) for x in midxs[1:end-1] for y in midys[1:end-1]]
#     sizes = [Vec2f0(midxs[i+1] - midxs[i], midys[j+1] - midys[j]) for i in eachindex(xs) for j in eachindex(ys)]

#     # zs need transpose to match previous result
#     s = meshscatter(pos, markersize=sizes, marker=m, color=zs'[:], shading=false)

#     return s
# end

# xs = tan.(atan(-3):0.1:atan(3))
# ys = -2.5:0.1:2.5 |> collect
# peaks(x,y) = 3*(1-x)^2 * exp(-(x^2) - (y+1)^2) - 10*(x/5 - x^3 - y^5) * exp(-x^2-y^2) - 1/3*exp(-(x+1)^2 - y^2)
# zs = [peaks(x,y) for x in xs, y in ys]
# stretched_heatmap(xs, ys, zs)
