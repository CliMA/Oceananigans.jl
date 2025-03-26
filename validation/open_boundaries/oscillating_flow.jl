# This validation script shows open boundaries working in a simple case where a flow past a 2D
# cylinder oscillates in two directions. All boundaries have the same
# `FlatExtrapolationOpenBoundaryCondition`s. This is similar to a more realistic case where we know
# some arbitary external conditions. First we test an xy flow and then we test an xz flow (the
# forcings and boundary conditions originally designed for `v` aere then used for `w` without
# modification).
#
# This case also has a stretched grid to validate the matching scheme on a stretched grid.

using Oceananigans, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition

@kwdef struct Cylinder{FT}
    D :: FT = 1.0
   x₀ :: FT = 0.0
   y₀ :: FT = 0.0
end

@inline (cylinder::Cylinder)(x, y) = ifelse((x - cylinder.x₀)^2 + (y - cylinder.y₀)^2 < (cylinder.D/2)^2, 1, 0)

architecture = CPU()

# model parameters
U = 1
D = 1.0
T = 50

cylinder = Cylinder(; D)

L = 10
Nx = Ny = 40

β = 0.2
x_faces(i) = L/2 * (β * ((2 * (i - 1)) / Nx - 1)^3 + (2 * (i - 1)) / Nx - 1) / (β + 1)
y = (-L/2, +L/2) .* D

xygrid = RectilinearGrid(architecture; topology = (Bounded, Bounded, Flat), size = (Nx, Ny), x = x_faces, y)
xzgrid = RectilinearGrid(architecture; topology = (Bounded, Flat, Bounded), size = (Nx, Ny), x = x_faces, z = y)

Δt = .5 * minimum_xspacing(xygrid) / abs(U)

@inline u∞(y, t, p) = p.U * cos(t * 2π / p.T) * (1 + 0.01 * randn())
@inline v∞(x, t, p) = p.U * sin(t * 2π / p.T) * (1 + 0.01 * randn())

function run_cylinder(grid, boundary_conditions; plot=true, stop_time = 50, simname = "")
    @info "Testing $simname with grid" grid

    cylinder_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)

    global model = NonhydrostaticModel(; grid,
                                       advection = UpwindBiased(order=5),
                                       forcing = (u = cylinder_forcing, v = cylinder_forcing, w = cylinder_forcing),
                                       boundary_conditions)

    @info "Constructed model"

    # initial noise to induce turbulance faster
    set!(model, u = U)

    @info "Set initial conditions"
    simulation = Simulation(model; Δt = Δt, stop_time = stop_time)

    # Callbacks
    wizard = TimeStepWizard(cfl = 0.1)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

    progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

    u, v, w = model.velocities
    filename = "cylinder_$(simname)_Nx_$Nx.jld2"

    if grid isa Oceananigans.Grids.ZFlatGrid
        outputs = (; model.velocities..., ζ = (@at (Center, Center, Center) ∂x(v) - ∂y(u)))
    elseif grid isa Oceananigans.Grids.YFlatGrid
        outputs = (; model.velocities..., ζ = (@at (Center, Center, Center) ∂x(w) - ∂z(u)))
    end
    simulation.output_writers[:velocity] = JLD2Writer(model, outputs,
                                                      overwrite_existing = true,
                                                      filename = filename,
                                                      schedule = TimeInterval(0.5),
                                                      with_halos = true)
    run!(simulation)

    if plot
        # load the results
        ζ_ts = FieldTimeSeries(filename, "ζ")
        u_ts = FieldTimeSeries(filename, "u")
        @info "Loaded results"

        xζ, yζ, zζ = nodes(ζ_ts, with_halos=true)
        xu, yu, zu = nodes(u_ts, with_halos=true)

        # plot the results
        fig = Figure(size = (600, 600))
        n = Observable(1)

        if grid isa Oceananigans.Grids.ZFlatGrid
            ζ_plt = @lift ζ_ts[:, :, 1, $n].parent
            ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "y", width = 400, height = 400, title = "ζ")
            heatmap!(ax, collect(xζ), collect(yζ), ζ_plt, colorrange = (-2, 2), colormap = :curl)

            u_plt = @lift u_ts[:, :, 1, $n].parent
            ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "y", width = 400, height = 400, title = "u")
            heatmap!(ax, collect(xu), collect(yu), u_plt, colorrange = (-2, 2), colormap = :curl)

        elseif grid isa Oceananigans.Grids.YFlatGrid
            ζ_plt = @lift ζ_ts[:, 1, :, $n].parent
            ax = Axis(fig[1, 1], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = 400, height = 400, title = "ζ")
            heatmap!(ax, collect(xζ), collect(zζ), ζ_plt, colorrange = (-2, 2), colormap = :curl)

            u_plt = @lift u_ts[:, 1, :, $n].parent
            ax = Axis(fig[1, 2], aspect = DataAspect(), xlabel = "x", ylabel = "z", width = 400, height = 400, title = "u")
            heatmap!(ax, collect(xu), collect(zu), u_plt, colorrange = (-2, 2), colormap = :curl)

        end
        resize_to_layout!(fig)
        record(fig, "ζ_$filename.mp4", 1:length(ζ_ts.times), framerate = 16) do i;
            n[] = i
            i % 10 == 0 && @info "$(n.val) of $(length(ζ_ts.times))"
        end
    end
end

matching_scheme_name(obc) = string(nameof(typeof(obc.classification.matching_scheme)))
for grid in (xygrid, xzgrid)

    u_fe = FlatExtrapolationOpenBoundaryCondition(u∞, parameters = (; U, T), relaxation_timescale = 1)
    v_fe = FlatExtrapolationOpenBoundaryCondition(v∞, parameters = (; U, T), relaxation_timescale = 1)
    w_fe = FlatExtrapolationOpenBoundaryCondition(v∞, parameters = (; U, T), relaxation_timescale = 1)

    u_boundaries_fe = FieldBoundaryConditions(west = u_fe, east = u_fe)
    v_boundaries_fe = FieldBoundaryConditions(south = v_fe, north = v_fe)
    w_boundaries_fe = FieldBoundaryConditions(bottom = w_fe, top = w_fe)

    if grid isa Oceananigans.Grids.ZFlatGrid
        boundary_conditions = (u = u_boundaries_fe, v = v_boundaries_fe)
        simname = "xy_" * matching_scheme_name(u_boundaries_fe.east)
    elseif grid isa Oceananigans.Grids.YFlatGrid
        boundary_conditions = (u = u_boundaries_fe, w = w_boundaries_fe)
        simname = "xz_" * matching_scheme_name(u_boundaries_fe.east)
    end
    run_cylinder(grid, boundary_conditions, simname = simname, stop_time = T)
end

