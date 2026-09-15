# # Global wind-driven gyres on a tripolar grid
#
# This example is a stripped-down version of the global ocean configurations used for
# OMIP-style simulations: a ¼° [`TripolarGrid`](@ref) with realistic coastlines,
# a ``z^\star`` vertical coordinate, and a [`SplitExplicitFreeSurface`](@ref).
# To keep it cheap enough for a laptop GPU, the ocean is a single 1-km-thick layer,
# which makes the model a nonlinear shallow water model on a sphere with continents.
#
# We force the ocean with an idealized zonal wind stress and look at the western boundary
# currents, the Gulf Stream and the Kuroshio, that close the wind-driven gyres.
# Sverdrup theory says that the depth-integrated meridional transport of the interior is
#
# ```math
# V = \frac{\boldsymbol{\hat z} \boldsymbol{\cdot} \boldsymbol{\nabla} \times \boldsymbol{\tau}}{\rho_o \beta} ,
# \qquad \beta = \frac{2 \Omega \cos \varphi}{R} ,
# ```
#
# and the western boundary current returns that transport back across the basin.
# Its strength should therefore scale with ``1 / \Omega``, which we check by running
# the simulation with three planetary rotation rates.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie, NCDatasets"
# ```

using Oceananigans
using Oceananigans.Units
using NCDatasets
using Downloads
using Printf
using CairoMakie
using CUDA # `using Metal` on Apple silicon

# We run on a GPU. Apple Metal GPUs only support single precision, so the floating
# point type is a parameter that Metal users should set to `Float32`.

arch = GPU()
FT = Float64
Oceananigans.defaults.FloatType = FT

# ## A single-layer tripolar grid
#
# The tripolar grid spans the globe from 80°S to the North Pole at ¼° resolution.
# The vertical direction has a single layer of depth `H`. We build it with a
# `MutableVerticalDiscretization` so that the layer thickness can follow the free surface,
# which is what the ``z^\star`` coordinate does.

Nx, Ny = 1440, 720
H = 1000 # layer depth [m]
z = MutableVerticalDiscretization((-H, 0))

underlying_grid = TripolarGrid(arch; size=(Nx, Ny, 1), z, halo=(5, 5, 5))

# ## Realistic coastlines from ETOPO1
#
# NOAA's ERDDAP server can subsample the 1-arc-minute ETOPO1 relief on the fly,
# so we download every 15th point: a ¼° map of the world of about 2 MB.

etopo_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?" *
            "altitude[(-89.875):15:(89.875)][(-179.875):15:(179.875)]"

etopo_filename = "etopo1_quarter_degree.nc"
isfile(etopo_filename) || Downloads.download(etopo_url, etopo_filename)

elevation = NCDataset(etopo_filename) do dataset
    nomissing(dataset["altitude"][:, :])
end

# We put the elevation on a `LatitudeLongitudeGrid` and interpolate it onto the tripolar grid,

etopo_grid = LatitudeLongitudeGrid(arch; size = (1440, 720),
                                   longitude = (-180, 180),
                                   latitude = (-90, 90),
                                   topology = (Periodic, Bounded, Flat))

etopo_elevation = CenterField(etopo_grid)
set!(etopo_elevation, elevation)

tripolar_elevation = Field{Center, Center, Nothing}(underlying_grid)
interpolate!(tripolar_elevation, etopo_elevation)

# and use it as a land mask: cells above sea level are land, everything else is
# a flat `H`-deep ocean. With a single layer, the bathymetry cannot vary, but
# the coastlines are what set the shape of the gyres.

bottom_height = Field{Center, Center, Nothing}(underlying_grid)
set!(bottom_height, ifelse.(interior(tripolar_elevation) .< 0, -H, 0))

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

# ## Wind stress and bottom drag
#
# The zonal wind stress
#
# ```math
# τˣ(φ) = - τ₀ \sin 2φ \sin 6φ
# ```
#
# has easterly trade winds peaking at ±15°, westerlies peaking at ±45°, and polar
# easterlies peaking at ±75°. Its curl drives cyclonic tropical and subpolar gyres and
# anticyclonic subtropical gyres, whose western boundary currents are the Gulf Stream
# and the Kuroshio. A positive flux boundary condition transports momentum out of the
# domain, so the momentum flux from the wind is minus the wind stress divided by
# the reference density `ρₒ`.

τ₀ = FT(0.15) # peak wind stress [N m⁻²]
ρₒ = 1020     # reference density [kg m⁻³]

zonal_wind_stress(φ, τ₀) = - τ₀ * sind(2φ) * sind(6φ)
zonal_momentum_flux(λ, φ, t, parameters) = - zonal_wind_stress(φ, parameters.τ₀) / parameters.ρₒ

# We plot the wind stress together with the Sverdrup transport per unit zonal width
# that it drives at Earth's rotation rate. The wind stress curl is
# ``- (1/R) \, \mathrm{d} τˣ / \mathrm{d} φ``.

R = Oceananigans.defaults.planet_radius
Ω = Oceananigans.defaults.planet_rotation_rate

wind_stress_curl(φ) = τ₀ / R * (2 * cosd(2φ) * sind(6φ) + 6 * sind(2φ) * cosd(6φ))
sverdrup_transport(φ, rotation_rate) = wind_stress_curl(φ) / (ρₒ * 2 * rotation_rate * cosd(φ) / R)

φ = -80:0.5:80

fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel="Latitude [°]", ylabel="Zonal wind stress [N m⁻²]")
lines!(ax, φ, zonal_wind_stress.(φ, τ₀))
ax = Axis(fig[1, 2], xlabel="Latitude [°]", ylabel="Sverdrup transport [m² s⁻¹]")
lines!(ax, φ[abs.(φ) .> 5], sverdrup_transport.(φ[abs.(φ) .> 5], Ω))
save("wind_stress.png", fig, px_per_unit=2) #hide

# ![](wind_stress.png)

# The wind stress enters through the top boundary condition on `u`, and a quadratic
# [`BulkDrag`](@ref) acts on the bottom.

wind_stress = FluxBoundaryCondition(zonal_momentum_flux, parameters=(; τ₀, ρₒ))
drag = BulkDrag(coefficient=FT(2.5e-3))
u_boundary_conditions = FieldBoundaryConditions(top=wind_stress, bottom=drag)
v_boundary_conditions = FieldBoundaryConditions(bottom=drag)

# ## Free surface and time stepping
#
# The barotropic gravity wave speed ``\sqrt{g H} ≈ 100`` m/s and the smallest ocean
# cell set the substep size of the split-explicit free surface. Given the time step
# `Δt`, the free surface computes the number of substeps that keeps the barotropic
# CFL number at 0.7.

Δt = 30minutes
free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=Δt)

# ## The model
#
# We use WENO advection schemes for momentum and for the tracer, and no explicit
# viscosity or diffusivity. The single layer has no vertical structure, so temperature
# is a passive tracer that shows how the gyres stir the surface ocean. It starts warm
# at the equator and cold at the poles.

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=7)

function build_model(grid, rotation_rate)
    coriolis = HydrostaticSphericalCoriolis(; rotation_rate)

    model = HydrostaticFreeSurfaceModel(grid; coriolis, free_surface,
                                        momentum_advection, tracer_advection,
                                        tracers = :T,
                                        buoyancy = nothing,
                                        vertical_coordinate = ZStarCoordinate(),
                                        boundary_conditions = (u=u_boundary_conditions, v=v_boundary_conditions))

    set!(model, T = (λ, φ, z) -> 30 * cosd(φ)^2)

    return model
end

# ## Running with three rotation rates
#
# The simulation runner saves the temperature and the barotropic streamfunction ``ψ``,
# defined by ``U = ∫ u \, \mathrm{d} z = - ∂ψ / ∂y`` and computed by integrating ``U``
# northward from Antarctica, every few days.

function run_gyres(grid, rotation_rate; stop_time=120days, save_interval=5days)
    model = build_model(grid, rotation_rate)
    simulation = Simulation(model; Δt, stop_time)

    wall_clock = Ref(time_ns())

    function progress(sim)
        u, v, w = sim.model.velocities
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("Ω = %.2e s⁻¹, iter: %d, time: %s, max|u|: %.2f m/s, wall time: %s",
                       rotation_rate, iteration(sim), prettytime(sim), maximum(abs, u), prettytime(elapsed))
        wall_clock[] = time_ns()
        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(500))

    u, v, w = model.velocities
    U = Field(Integral(u, dims=3))
    ψ = Field(CumulativeIntegral(-U, dims=2))
    T = model.tracers.T

    filename = @sprintf("global_wind_driven_gyres_%.2e.jld2", rotation_rate)

    simulation.output_writers[:surface] = JLD2Writer(model, (; T, ψ); filename,
                                                     schedule = TimeInterval(save_interval),
                                                     array_type = Array{Float32},
                                                     overwrite_files = true)

    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
    run!(simulation)

    return filename
end

rotation_rates = (Ω / 2, Ω, 2Ω)
filenames = Dict(rotation_rate => run_gyres(grid, rotation_rate) for rotation_rate in rotation_rates)

# ## Gyre transports
#
# The transport of a gyre is the difference between the streamfunction at the gyre
# center and on the coast, so we measure it as the range of ``ψ`` over a box that
# contains the center of the subtropical gyre and the western boundary. The Gulf Stream
# and the Kuroshio carry that transport northward along the coast. The tripolar grid's
# longitude runs from 70°E eastward around the globe, so we wrap it to ``[0, 360)``.

ψt = FieldTimeSeries(filenames[Ω], "ψ")
times = ψt.times
λ = λnodes(ψt.grid, Face(), Center(), Center())
φ = φnodes(ψt.grid, Face(), Center(), Center())

function gyre_transport(ψ, box)
    inside = @. (box.longitude[1] < mod(λ, 360) < box.longitude[2]) & (box.latitude[1] < φ < box.latitude[2])
    values = interior(ψ, :, :, 1)[inside]
    return maximum(values) - minimum(values)
end

gulf_stream = (longitude = (275, 310), latitude = (20, 42))
kuroshio = (longitude = (118, 160), latitude = (20, 42))

# For the Sverdrup prediction, we integrate the interior transport across each basin
# on the ETOPO grid and take the largest value over the latitudes of the gyre.

etopo_longitude = mod.(-179.875:0.25:179.875, 360)
etopo_latitude = -89.875:0.25:89.875
ocean = elevation .< 0

function sverdrup_gyre_transport(rotation_rate, basin)
    transports = map(basin.latitude[1]:0.25:basin.latitude[2]) do latitude
        j = argmin(abs.(etopo_latitude .- latitude))
        inside = @. basin.longitude[1] < etopo_longitude < basin.longitude[2]
        width = count(ocean[:, j] .& inside) * 2π * R * cosd(latitude) / length(etopo_longitude)
        return - sverdrup_transport(latitude, rotation_rate) * width
    end
    return maximum(transports)
end

atlantic = (longitude = (280, 360), latitude = (20, 42))
pacific = (longitude = (120, 250), latitude = (20, 42))

# Now we compare the transport time series with the Sverdrup prediction, which
# halves every time the rotation rate doubles.

Sv = 1e6 # m³ s⁻¹

fig = Figure(size=(900, 400))
axes = (gulf_stream = Axis(fig[1, 1], title="Gulf Stream", xlabel="Time [days]", ylabel="Transport [Sv]"),
        kuroshio = Axis(fig[1, 2], title="Kuroshio", xlabel="Time [days]"))

colors = Dict(zip(rotation_rates, Makie.wong_colors()))

for rotation_rate in rotation_rates
    streamfunctions = FieldTimeSeries(filenames[rotation_rate], "ψ")
    label = @sprintf("Ω = %.1f Ω_Earth", rotation_rate / Ω)
    color = colors[rotation_rate]

    for (name, box, basin) in ((:gulf_stream, gulf_stream, atlantic), (:kuroshio, kuroshio, pacific))
        transport = [gyre_transport(streamfunctions[n], box) for n in 1:length(times)]
        lines!(axes[name], times / day, transport / Sv; label, color)
        hlines!(axes[name], sverdrup_gyre_transport(rotation_rate, basin) / Sv; color, linestyle=:dash)
    end
end

axislegend(axes.gulf_stream, position=:rt)
save("western_boundary_current_transports.png", fig, px_per_unit=2) #hide

# ![](western_boundary_current_transports.png)
#
# The solid lines are the gyre transports measured in the simulations and the dashed
# lines the Sverdrup prediction. Doubling the rotation rate halves the transport of
# both boundary currents.
#
# ## The gyres
#
# Finally we plot the streamfunction at the end of each simulation. The Antarctic
# Circumpolar Current puts a large offset between ``ψ`` on Antarctica and everywhere
# else, so we set ``ψ = 0`` on North America. The tripolar grid is curvilinear, so we
# draw the fields with `surface!` on the grid's own longitudes and latitudes, and hide
# the land.

land = Array(interior(bottom_height, :, :, 1)) .≥ 0
north_america = argmin(@. (mod(λ, 360) - 260)^2 + (φ - 40)^2)
longitude_ticks = (120:60:420, ["120°E", "180°", "120°W", "60°W", "0°", "60°E"])

function map_axis(figure_position; title="")
    return Axis(figure_position; title, xlabel="Longitude", ylabel="Latitude",
                aspect=DataAspect(), limits=((70, 430), (-80, 70)), xticks=longitude_ticks)
end

fig = Figure(size=(900, 1000))

for (row, rotation_rate) in enumerate(rotation_rates)
    streamfunctions = FieldTimeSeries(filenames[rotation_rate], "ψ")
    ψ_end = interior(streamfunctions[end], :, :, 1)
    streamfunction = ifelse.(land, NaN, (ψ_end .- ψ_end[north_america]) / Sv)
    axis = map_axis(fig[row, 1]; title=@sprintf("Ω = %.1f Ω_Earth", rotation_rate / Ω))
    sf = surface!(axis, λ, φ, 0 * λ; color=streamfunction, colormap=:balance, colorrange=(-100, 100),
                  shading=NoShading, nan_color=:gray)
    row == 1 && Colorbar(fig[1:length(rotation_rates), 2], sf, label="Streamfunction [Sv]")
end

save("global_wind_driven_gyres.png", fig, px_per_unit=2) #hide

# ![](global_wind_driven_gyres.png)
#
# The temperature tracer shows the gyres at work: the boundary currents carry warm
# water poleward along the western coasts and the subpolar gyres bring cold water south.

Tt = FieldTimeSeries(filenames[Ω], "T")
n = Observable(1)
temperature = @lift ifelse.(land, NaN, interior(Tt[$n], :, :, 1))
title = @lift "Temperature after " * prettytime(times[$n])

fig = Figure(size=(900, 500))
ax = map_axis(fig[1, 1]; title)
sf = surface!(ax, λ, φ, 0 * λ; color=temperature, colormap=:thermal, colorrange=(0, 30),
              shading=NoShading, nan_color=:gray)
Colorbar(fig[1, 2], sf, label="Temperature [°C]")

record(fig, "global_wind_driven_gyres.mp4", 1:length(times), framerate=8) do frame
    n[] = frame
end
nothing #hide

# ![](global_wind_driven_gyres.mp4)
