# Validation script for Stevens (1990) open boundary conditions.
#
# A 2D x-z channel with uniform outflow velocity U₀. A smooth Gaussian tracer
# blob and a Gaussian free surface bump are initialized. Stevens OBCs radiate
# the tracer outward, PerturbationAdvection handles velocity, and a sponge layer
# damps free surface gravity waves near boundaries. Produces a video.

using Oceananigans
using Oceananigans.BoundaryConditions: StevensAdvection, PerturbationAdvection
using Printf
using CairoMakie

architecture = CPU()

Nx, Nz = 128, 16
Lx, Lz = 1.0, 0.25

grid = RectilinearGrid(architecture;
                        topology = (Bounded, Flat, Bounded),
                        size = (Nx, Nz),
                        x = (0, Lx),
                        z = (-Lz, 0))

U₀ = 0.1
T_bg = 10.0

# PerturbationAdvection for velocity: lets perturbations (gravity waves) exit
pa = PerturbationAdvection(outflow_timescale = Inf, inflow_timescale = 0)
u_east_bc = OpenBoundaryCondition(U₀; scheme = pa)
u_west_bc = OpenBoundaryCondition(U₀; scheme = pa)

# Stevens for tracers: radiation with phase velocity and relaxation on inflow
stevens_tracer = StevensAdvection(relaxation_timescale = 10.0, use_phase_velocity = true)
T_east_bc = OpenBoundaryCondition(T_bg; scheme = stevens_tracer)
T_west_bc = OpenBoundaryCondition(T_bg; scheme = stevens_tracer)

u_bcs = FieldBoundaryConditions(east = u_east_bc, west = u_west_bc)
T_bcs = FieldBoundaryConditions(east = T_east_bc, west = T_west_bc)

model = HydrostaticFreeSurfaceModel(grid;
                                     free_surface = ImplicitFreeSurface(),
                                     tracers = :T,
                                     buoyancy = nothing,
                                     boundary_conditions = (u = u_bcs, T = T_bcs))

# Smooth Gaussian tracer blob
x_tracer = 0.25 * Lx
σ_tracer = Lx / 12
ΔT = 5.0
Tᵢ(x, z) = T_bg + ΔT * exp(-(x - x_tracer)^2 / (2 * σ_tracer^2))

# Gaussian free surface bump
x_eta = 0.5 * Lx
σ_eta = Lx / 10
η₀ = 0.005
ηᵢ(x, z) = η₀ * exp(-(x - x_eta)^2 / (2 * σ_eta^2))

set!(model, u = U₀, T = Tᵢ, η = ηᵢ)

Δt = 0.3 * (Lx / Nx) / U₀
stop_time = 1.5 * Lx / U₀
simulation = Simulation(model; Δt, stop_time)

# Sponge layer: damps free surface η near east/west boundaries.
# The implicit solver computes η each step; the sponge modifies it afterwards.
# On the next step, the solver uses the modified η as the "old" value.
sponge_width = 20  # cells
sponge_mask = ones(Nx)
for i in 1:sponge_width
    α = 0.5 * (1 - cos(π * i / sponge_width))  # smooth: 0 at boundary, 1 at interior
    sponge_mask[i] = α
    sponge_mask[Nx - i + 1] = α
end

function apply_sponge!(sim)
    η_interior = interior(sim.model.free_surface.displacement, :, 1, 1)
    for i in 1:Nx
        @inbounds η_interior[i] *= sponge_mask[i]
    end
end

simulation.callbacks[:sponge] = Callback(apply_sponge!, IterationInterval(1))

# Collect snapshots
n_frames = 200
save_interval = stop_time / n_frames

T_snapshots = []
η_snapshots = []
t_snapshots = Float64[]

function save_snapshot!(sim)
    push!(T_snapshots, Array(interior(sim.model.tracers.T, :, 1, :)))
    push!(η_snapshots, Array(interior(sim.model.free_surface.displacement, :, 1, 1)))
    push!(t_snapshots, time(sim))
end

simulation.callbacks[:save] = Callback(save_snapshot!, TimeInterval(save_interval))
save_snapshot!(simulation)

progress(sim) = @printf("Iteration %d, time = %.3f, max|T-Tbg| = %.4f, max|η| = %.2e\n",
                         iteration(sim), time(sim),
                         maximum(abs, interior(sim.model.tracers.T) .- T_bg),
                         maximum(abs, interior(sim.model.free_surface.displacement)))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

@info "Running Stevens OBC validation..."
run!(simulation)

# ---- Video production ----
@info "Producing video from $(length(t_snapshots)) frames..."

xc = xnodes(model.tracers.T)
zc = znodes(model.tracers.T)

fig = Figure(size = (900, 700))

ax_eta = Axis(fig[1, 1];
              xlabel = "x (m)",
              ylabel = "η (m)",
              title = "Free surface elevation")
ylims!(ax_eta, -1.5 * η₀, 2.0 * η₀)

ax_T = Axis(fig[2, 1];
            xlabel = "x (m)",
            ylabel = "z (m)",
            title = "Tracer T(x, z)")

time_label = Label(fig[0, 1], "t = 0.00 s", fontsize = 20, tellwidth = false)

eta_line = lines!(ax_eta, xc, η_snapshots[1]; color = :dodgerblue, linewidth = 2)
hlines!(ax_eta, [0.0]; color = :gray, linestyle = :dash, linewidth = 0.5)

hm = heatmap!(ax_T, xc, zc, T_snapshots[1];
              colormap = :thermal,
              colorrange = (T_bg - 0.5, T_bg + ΔT))
Colorbar(fig[2, 2], hm; label = "T")

video_path = joinpath(@__DIR__, "stevens_open_boundary.mp4")

record(fig, video_path, eachindex(t_snapshots); framerate = 24) do idx
    eta_line[2] = η_snapshots[idx]
    hm[3] = T_snapshots[idx]
    time_label.text[] = @sprintf("t = %.2f s", t_snapshots[idx])
end

@info "Video saved to $video_path"
