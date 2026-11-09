# Dense water overflow down a continental slope (Haidvogel and Beckmann, 1999; Ilicak et al., 2012) at a
# resolution typical of a coarse ocean model, comparing GridFittedBottom, PartialCellBottom and ShavedCellBottom
# on a z-star vertical coordinate.

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: inactive_node, rnode
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node
using Oceananigans.ImmersedBoundaries: static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ
using CairoMakie
using Printf

Lx = 200kilometers
H  = 2000meters
Nx = 50
Nz = 20

shelf_depth = 500meters
slope_start = 20kilometers
slope_end   = 60kilometers

function bathymetry(x)
    ξ = clamp((x - slope_start) / (slope_end - slope_start), 0, 1)
    return - shelf_depth - (H - shelf_depth) * ξ
end

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (0, Lx),
                                  z = MutableVerticalDiscretization((-H, 0)),
                                  topology = (Bounded, Flat, Bounded))

bottoms = (full = GridFittedBottom(bathymetry),
           partial = PartialCellBottom(bathymetry),
           shaved = ShavedCellBottom(bathymetry; minimum_fractional_cell_height=0.001))

thermal_expansion = 2e-4
reference_density = 1027
equation_of_state = LinearEquationOfState(; thermal_expansion, haline_contraction=0)
buoyancy = SeawaterBuoyancy(; equation_of_state, constant_salinity=35)

dense_temperature   = 0
ambient_temperature = 10
initial_temperature(x, z) = x < slope_start ? dense_temperature : ambient_temperature

stop_time = 1day
save_interval = 10minutes

function run_overflow(name, bottom)
    grid = ImmersedBoundaryGrid(underlying_grid, bottom)

    model = HydrostaticFreeSurfaceModel(grid; buoyancy,
                                        tracers = :T,
                                        momentum_advection = WENO(order=5, minimum_buffer_upwind_order=1, time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)),
                                        tracer_advection = WENO(order=5, bounds=(dense_temperature, ambient_temperature), time_discretization = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)),
                                        closure = VerticalScalarDiffusivity(ν=1e-4),
                                        free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                                        timestepper = :SplitRungeKutta3)

    set!(model, T=initial_temperature)

    simulation = Simulation(model; Δt=2minutes, stop_time)

    progress(sim) = @info @sprintf("%7s | iter %4d | t = %s | max|u| = %.2e m s⁻¹ | max|w| = %.2e m s⁻¹", name,
                                   iteration(sim), prettytime(sim), maximum(abs, sim.model.velocities.u),
                                   maximum(abs, sim.model.velocities.w))

    add_callback!(simulation, progress, TimeInterval(2hours))

    u, v, w = model.velocities
    ρ = reference_density * (1 - thermal_expansion * (model.tracers.T - ambient_temperature))
    ω = ∂z(u) - ∂x(w)

    simulation.output_writers[:fields] = JLD2Writer(model, (; ρ, ω, w);
                                                    filename = "shaved_cell_overflow_$name.jld2",
                                                    schedule = TimeInterval(save_interval),
                                                    overwrite_files = true)

    run!(simulation)

    return grid
end

grids = NamedTuple(name => run_overflow(name, bottom) for (name, bottom) in pairs(bottoms))

#####
##### Bathymetry as the model sees it
#####

xᶠ = xnodes(underlying_grid, Face())
xᶜ = xnodes(underlying_grid, Center())
top = rnode(1, 1, Nz+1, underlying_grid, Center(), Center(), Face())

center_bottom(grid) = [top - static_column_depthᶜᶜᵃ(i, 1, grid) for i in 1:Nx]

# Full and partial cells are flat across each column.
function bottom_outline(grid)
    zᵇ = center_bottom(grid)
    x = vec(permutedims(hcat(xᶠ[1:Nx], xᶠ[2:Nx+1])))
    z = repeat(zᵇ, inner=2)
    return x, z
end

# Shaved cells follow the piecewise-linear surface through their faces.
bottom_outline(grid::ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:ShavedCellBottom}) = xᶠ, [top - static_column_depthᶠᶜᵃ(i, 1, grid) for i in 1:Nx+1]

function solid_polygon(grid)
    x, z = bottom_outline(grid)
    return Point2f.(vcat(x, reverse(x)) ./ 1e3, vcat(z, fill(-H, length(x))))
end

#####
##### Animation
#####

titles = (full = "full cells", partial = "partial cells", shaved = "shaved cells")

timeseries = NamedTuple(name => NamedTuple(variable => FieldTimeSeries("shaved_cell_overflow_$name.jld2", string(variable))
                                           for variable in (:ρ, :ω, :w))
                        for name in keys(bottoms))
times = timeseries.full.ρ.times

function masked(field_timeseries, n)
    data = copy(interior(field_timeseries[n], :, 1, :))
    grid = field_timeseries.grid
    ℓx, _, ℓz = Oceananigans.Fields.instantiated_location(field_timeseries)
    vorticity_node = ℓx isa Face && ℓz isa Face
    for i in axes(data, 1), k in axes(data, 2)
        # ∂z(u) next to the bottom differences against the u zeroed on the free-slip boundary
        wall = vorticity_node && (immersed_peripheral_node(i, 1, k-1, grid, Face(), Center(), Center()) || immersed_peripheral_node(i, 1, k, grid, Face(), Center(), Center()))
        (inactive_node(i, 1, k, grid, ℓx, Center(), ℓz) || wall) && (data[i, k] = NaN)
    end
    return data
end

variables = (ρ = (label = "ρ (kg m⁻³)", colormap = Reverse(:deep), colorrange = (1027, 1029)),
             ω = (label = "∂u/∂z - ∂w/∂x (s⁻¹)", colormap = :balance, colorrange = (-2e-2, 2e-2)),
             w = (label = "w (m s⁻¹)", colormap = :balance, colorrange = (-0.1, 0.1)))

n = Observable(1)

fig = Figure(size = (1800, 1000))
Label(fig[0, 1:3], @lift(@sprintf("Overflow, Δx = %d km, Δz = %d m, t = %s", Lx / Nx / 1e3, H / Nz, prettytime(times[$n]))),
      fontsize = 22, tellwidth = false)

for (row, variable) in enumerate(keys(variables))
    attributes = variables[variable]
    local heatmap_plot

    for (column, name) in enumerate(keys(bottoms))
        field_timeseries = timeseries[name][variable]
        ℓx, _, ℓz = Oceananigans.Fields.instantiated_location(field_timeseries)
        x = xnodes(underlying_grid, ℓx) ./ 1e3
        z = rnodes(underlying_grid, ℓz)

        ax = Axis(fig[row, column],
                  title = row == 1 ? titles[name] : "",
                  xlabel = row == 3 ? "x (km)" : "",
                  ylabel = column == 1 ? "z (m)" : "",
                  limits = ((0, Lx / 1e3), (-H, 0)))

        row < 3 && hidexdecorations!(ax, grid=false)
        column > 1 && hideydecorations!(ax, grid=false)

        data = @lift masked(field_timeseries, $n)
        heatmap_plot = heatmap!(ax, x, z, data; colormap=attributes.colormap, colorrange=attributes.colorrange,
                                nan_color=:transparent)

        poly!(ax, solid_polygon(grids[name]), color=:gray70, strokecolor=:black, strokewidth=1.5)
        lines!(ax, xᶜ ./ 1e3, bathymetry.(xᶜ), color=:black, linestyle=:dash, linewidth=1)
    end

    Colorbar(fig[row, 4], heatmap_plot, label=attributes.label)
end

record(fig, "shaved_cell_overflow.mp4", 1:length(times), framerate=12) do nn
    n[] = nn
end

save("shaved_cell_overflow.png", fig)
