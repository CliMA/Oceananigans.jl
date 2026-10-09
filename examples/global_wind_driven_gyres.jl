# # Global wind-driven gyres on a tripolar grid
#
# This example is a stripped-down version of the global ocean configurations used for
# OMIP-style simulations: a [`TripolarGrid`](@ref) with realistic bathymetry,
# a [``z^\star`` vertical coordinate](@ref "Vertical coordinates"),
# and a [`SplitExplicitFreeSurface`](@ref).
# To keep it cheap enough for a laptop GPU, the grid is 1° with four layers.
#
# We force the ocean with an idealized zonal wind stress and look at the western boundary
# currents, the Gulf Stream, and the Kuroshio, that close the wind-driven gyres.
# Sverdrup theory says that the depth-integrated meridional transport of the interior is
#
# ```math
# V = \frac{\boldsymbol{\hat z} \boldsymbol{\cdot} (\boldsymbol{\nabla} \times \boldsymbol{\tau})}{\rho₀ \beta} ,
# \qquad \beta = \frac{2 \Omega \cos \varphi}{R} ,
# ```
#
# and the western boundary current returns that transport back across the basin.
# Its strength should therefore scale with ``1 / \Omega``, which we check by running
# the simulation at Earth's rotation rate and at half that. A third run with a constant
# Coriolis parameter shows that the gyres owe their western intensification to ``β``.
#
# ## Install dependencies
#
# Besides Oceananigans, this example needs NCDatasets to read the bathymetry, CairoMakie
# to make the plots, and ConservativeRegridding to regrid between the tripolar and
# latitude-longitude grids. To run on a GPU you also need the Julia package for your GPU:
# CUDA for NVIDIA, Metal for Apple silicon, or AMDGPU for AMD. On a machine without a GPU,
# skip that package and the example runs on the CPU.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, CairoMakie, ConservativeRegridding"
# pkg"add CUDA" # or Metal, or AMDGPU
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Coriolis: DualGridScheme
using Oceananigans.Grids: φnode
using Oceananigans.ImmersedBoundaries: InterfaceImmersedCondition
using NCDatasets
using Printf
using CairoMakie
using ConservativeRegridding

# We run on the GPU if there is one: Metal on Apple silicon, CUDA if an NVIDIA driver is
# installed, AMDGPU if a ROCm driver is installed, and the CPU otherwise.
#
# To keep the demo fast, everything runs in single precision (`Float32`), which is much faster
# than double precision on most GPUs. For research-grade simulations we recommend `Float64`,
# which Metal does not currently support, so Apple GPUs are limited to `Float32`.

arch = if Sys.isapple() && Sys.ARCH === :aarch64
    using Metal
    Metal.functional() ? GPU(Metal.MetalBackend()) : CPU()
elseif !isnothing(Sys.which("nvidia-smi"))
    using CUDA
    CUDA.functional() ? GPU() : CPU()
elseif !isnothing(Sys.which("rocminfo"))
    using AMDGPU
    AMDGPU.functional() ? GPU(AMDGPU.ROCBackend()) : CPU()
else
    CPU()
end

FT = Float32
Oceananigans.defaults.FloatType = FT

# ## A four-layer tripolar grid
#
# The tripolar grid spans the globe from 80°S to the North Pole. The resolution is a
# parameter: the three six-year 1° runs below take just under two hours on a laptop
# GPU, ½° takes about eight times longer, and 2° is quick enough for a CPU. The four layers thicken
# with depth, from 100 m at the surface to 2.5 km at the bottom. We build the vertical
# coordinate with a `MutableVerticalDiscretization` so that the layers can stretch with
# the free surface, which is what the ``z^\star`` coordinate does.

resolution = 1 # degrees
Nx = round(Int, 360 / resolution)
Ny = Nx ÷ 2
z_faces = [-4000, -1500, -500, -100, 0]
Nz = length(z_faces) - 1
z = MutableVerticalDiscretization(z_faces)

underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), z, halo=(5, 5, 5))

# ## Bathymetry from ETOPO1
#
# NOAA's NCEI serves the 1-arc-minute ETOPO1 relief over OPeNDAP, so NCDatasets can read
# every `stride`-th point, three per grid cell, without downloading the whole file.
# Averaging the samples in 3 × 3 blocks then gives the mean elevation of each grid cell
# rather than the elevation at a single point.

stride = round(Int, 20resolution) # arc-minutes between samples
etopo_url = "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO1_Ice_g_gmt4.nc"

etopo_elevation = NCDataset(etopo_url) do dataset
    i = 1 + stride÷2 : stride : dataset.dim["lon"] - stride÷2
    j = 1 + stride÷2 : stride : dataset.dim["lat"] - stride÷2
    nomissing(dataset["z"][i, j])
end

block_mean(a, n) = [sum(@view a[i:i+n-1, j:j+n-1]) / n^2 for i in 1:n:size(a, 1), j in 1:n:size(a, 2)]
elevation = FT.(block_mean(etopo_elevation, 3))
nothing #hide

# We put the elevation on a `LatitudeLongitudeGrid`, interpolate it onto the tripolar
# grid, and use it as the bottom height of a `GridFittedBottom`. With the
# `InterfaceImmersedCondition`, a cell is land only when it lies entirely below the sea
# floor, so the ocean is 100, 500, 1500, or 4000 m deep: the shelf seas and straits stay
# open at 100 m and ridges deeper than 1500 m are flattened to 4000 m.

etopo_grid = LatitudeLongitudeGrid(arch; size = size(elevation),
                                   longitude = (-180, 180),
                                   latitude = (-90, 90),
                                   topology = (Periodic, Bounded, Flat))

elevation_field = CenterField(etopo_grid)
set!(elevation_field, elevation)

bottom_height = Field{Center, Center, Nothing}(underlying_grid)
interpolate!(bottom_height, elevation_field)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height, InterfaceImmersedCondition()); active_cells_map=true)

# The Makie extension's `quadmesh!` draws each cell of the bathymetry field as a flat-colored
# quadrilateral in the tripolar grid's own longitudes and latitudes, which run from 70°E
# eastward around the globe. Since the field lives on the immersed grid, land is masked
# automatically.

height = bottom_height_field(grid)
bathymetry = Field{Center, Center, Nothing}(grid)
set!(bathymetry, -height)

longitude_ticks = (120:60:420, ["120°E", "180°", "120°W", "60°W", "0°", "60°E"])

map_axis(fig_position; title="", limits=((70, 430), (-80, 70)), xticks=longitude_ticks) =
    Axis(fig_position; title, limits, xticks, aspect=DataAspect(),
                       xlabel="Longitude", ylabel="Latitude")

fig = Figure(size=(900, 500))
ax = map_axis(fig[1, 1]; title="Ocean depth")
qm = quadmesh!(ax, bathymetry; colormap=:deep, nan_color=:gray)
Colorbar(fig[1, 2], qm, label="Depth [m]")
save("bathymetry.png", fig, px_per_unit=2) #hide

# ![](bathymetry.png)

# ## Wind stress and bottom drag
#
# The zonal wind stress is a sum of four Gaussian wind belts,
#
# ```math
# τˣ(φ) = \sum_n τₙ \exp\left[-\left(\frac{φ - φₙ}{11°}\right)^2\right] ,
# ```
#
# the easterly trade winds and the westerlies of each hemisphere, whose strengths and latitudes
# follow the observed annual- and zonal-mean wind stress over the ocean from the NCEP/NCAR
# reanalysis. The trades overlap into weak easterlies on the equator, and the westerlies are
# almost three times stronger over the Southern Ocean than in the north. The curl drives cyclonic
# tropical and subpolar gyres and anticyclonic subtropical gyres, whose western boundary
# currents are the Gulf Stream and the Kuroshio. A positive flux boundary condition transports
# momentum out of the domain, so the momentum flux from the wind is minus the wind stress
# divided by the reference density `ρ₀`.

wind_belts = (northern_trades = FT(-0.055), southern_trades = FT(-0.065),
              northern_westerlies = FT(0.07), southern_westerlies = FT(0.19)) # [N m⁻²]
ρ₀ = 1020 # reference density [kg m⁻³]

wind_belt(φ, τ, φ₀) = τ * exp(-((φ - φ₀) / 11)^2)

zonal_wind_stress(φ, belts) = wind_belt(φ, belts.northern_trades, 18) + wind_belt(φ, belts.southern_trades, -17) +
                              wind_belt(φ, belts.northern_westerlies, 46) + wind_belt(φ, belts.southern_westerlies, -51)

zonal_momentum_flux(λ, φ, t, parameters) = - zonal_wind_stress(φ, parameters.wind_belts) / parameters.ρ₀

# We plot the wind stress together with the Sverdrup transport per unit zonal width
# that it drives at Earth's rotation rate. The wind stress curl is
# ``- R^{-1} \, ∂τˣ / ∂φ``.

R = Oceananigans.defaults.planet_radius
Ω = Oceananigans.defaults.planet_rotation_rate

wind_stress_curl(φ) = - (zonal_wind_stress(φ + 0.01, wind_belts) - zonal_wind_stress(φ - 0.01, wind_belts)) / (R * deg2rad(0.02))
sverdrup_transport(φ, rotation_rate) = wind_stress_curl(φ) / (ρ₀ * 2 * rotation_rate * cosd(φ) / R)

latitudes = -80:0.5:80

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1], xlabel="Zonal wind stress [N m⁻²]", ylabel="Latitude [°]")
lines!(ax, [zonal_wind_stress(φ, wind_belts) for φ in latitudes], latitudes)
ax = Axis(fig[1, 2], xlabel="Sverdrup transport [m² s⁻¹]", ylabel="Latitude [°]")
lines!(ax, sverdrup_transport.(latitudes[abs.(latitudes) .> 5], Ω), latitudes[abs.(latitudes) .> 5])
save("wind_stress.png", fig, px_per_unit=2) #hide

# ![](wind_stress.png)

# The wind stress enters through the top boundary condition on `u`. A quadratic
# [`BulkDrag`](@ref) acts on the sea floor, which is the bottom of the domain in the
# deep ocean and an immersed boundary everywhere else.

wind_stress = FluxBoundaryCondition(zonal_momentum_flux, parameters=(; wind_belts, ρ₀))
drag = BulkDrag(coefficient=FT(2.5e-3))
u_boundary_conditions = FieldBoundaryConditions(top=wind_stress, bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
v_boundary_conditions = FieldBoundaryConditions(bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))

# ## Surface temperature restoring
#
# The surface layer is restored on a 30-day time scale to ``T^\star(φ) = 30 \cos^2 φ``,
# warm at the equator and cold at the poles. The flux is written in discrete form so that
# it can read the surface temperature at each column.

restoring_temperature(φ) = 30 * cos(deg2rad(φ))^2

@inline function temperature_flux(i, j, grid, clock, fields, parameters)
    φ = φnode(i, j, grid.Nz, grid, Center(), Center(), Center())
    return @inbounds parameters.rate * (fields.T[i, j, grid.Nz] - restoring_temperature(φ))
end

restoring_rate = FT(100 / 30days) # surface layer thickness over the restoring time scale [m s⁻¹]
temperature_restoring = FluxBoundaryCondition(temperature_flux; discrete_form=true, parameters=(; rate=restoring_rate))
T_boundary_conditions = FieldBoundaryConditions(top=temperature_restoring)

# ## Free surface and time stepping
#
# The barotropic gravity wave speed ``\sqrt{g H} ≈ 200`` m/s and the smallest ocean
# cell set the substep size of the split-explicit free surface.
# Given the time step `Δt`, the free surface computes the number of substeps that keeps
# the barotropic CFL number at 0.7.

Δt = 1hour
free_surface = SplitExplicitFreeSurface(grid; cfl=0.7, fixed_Δt=Δt)

# ## The model
#
# We use WENO advection schemes for momentum and for temperature. Temperature sets the
# buoyancy through a linear equation of state. It starts from a horizontally uniform
# exponential thermocline with a 1 kilometer scale: a meridional density gradient across
# a basin comes with a depth-integrated thermal wind of hundreds of Sverdrups that would
# swamp the wind-driven gyres and take years to adjust away, so the equator-to-pole
# contrast enters only through the surface restoring.

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=7)
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4), constant_salinity=35)

# Poleward of about 60°, the restoring cools the surface below the water underneath it.
# A hydrostatic model cannot overturn such a column, so a convective adjustment mixes
# temperature and momentum vertically wherever the stratification is unstable.

closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz=1, convective_νz=1)

function build_model(grid, coriolis)
    model = HydrostaticFreeSurfaceModel(grid; coriolis, free_surface, buoyancy, closure,
                                        momentum_advection, tracer_advection,
                                        tracers = :T,
                                        timestepper = :SplitRungeKutta3,
                                        vertical_coordinate = ZStarCoordinate(),
                                        boundary_conditions = (u=u_boundary_conditions, v=v_boundary_conditions, T=T_boundary_conditions))

    set!(model, T = (λ, φ, z) -> 10 * exp(z / 1000))

    return model
end

# ## Three Coriolis parameters
#
# The Coriolis parameter is the only thing that differs between the runs. Two of them use
# the spherical ``f = 2Ω \sin φ``, at Earth's rotation rate and at half it, which halves
# ``β`` and should double the Sverdrup transport. The third uses an [`FPlane`](@ref) with the
# value of ``f`` at 30°N everywhere, ``f = 2Ω \sin 30°``, so that ``β = 0``. A constant
# ``f`` has the wrong sign in the Southern Hemisphere, so on the ``f``-plane we only look
# at the northern gyres.
#
# The simulation runner saves the barotropic streamfunction ``ψ``, defined by
# ``U = ∫ u \, \mathrm{d} z = - ∂ψ / ∂y`` and computed by integrating ``U`` northward from
# Antarctica, together with the surface speed and the surface temperature, every ten days
# of a six-year run.

year = 365days

function run_gyres(grid, coriolis, name; stop_time=6year, save_interval=10days)
    model = build_model(grid, coriolis)
    simulation = Simulation(model; Δt, stop_time)

    wall_clock = Ref(time_ns())

    function progress(sim)
        u, v, w = sim.model.velocities
        elapsed = 1e-9 * (time_ns() - wall_clock[])
        @info @sprintf("%s, iter: %d, time: %s, max|u|: %.2f m/s, wall time: %s",
                       name, iteration(sim), prettytime(sim), maximum(abs, u), prettytime(elapsed))
        wall_clock[] = time_ns()
        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(500))

    u, v, w = model.velocities
    U = Field(Integral(u, dims=3))
    ψ = Field(CumulativeIntegral(-U, dims=2))
    speed = Field(@at (Center, Center, Center) sqrt(u^2 + v^2))
    surface_speed = view(speed, :, :, grid.Nz)
    surface_temperature = view(model.tracers.T, :, :, grid.Nz)

    filename = "global_wind_driven_gyres_$name.jld2"

    simulation.output_writers[:surface] = JLD2Writer(model, (; ψ, surface_speed, surface_temperature); filename,
                                                     schedule = TimeInterval(save_interval),
                                                     array_type = Array{Float32},
                                                     overwrite_files = true)

    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
    run!(simulation)

    return filename
end

rotation_rates = (Ω, Ω / 2)
coriolis_titles = Dict(Ω => "f = 2Ω sin φ", Ω / 2 => "f = Ω sin φ")
filenames = Dict(rotation_rate => run_gyres(grid, HydrostaticSphericalCoriolis(; rotation_rate, scheme=DualGridScheme(grid)), @sprintf("omega_%g", rotation_rate / Ω))
                 for rotation_rate in rotation_rates)

f_plane_filename = run_gyres(grid, FPlane(latitude=30, scheme=DualGridScheme(grid)), "f_plane")

# ## Gyre transports
#
# The transport of a gyre is the difference between the streamfunction at the gyre
# center and on the coast, so we measure it as the range of ``ψ`` over a box that
# contains the center of the subtropical gyre and the western boundary. The Gulf Stream
# and the Kuroshio carry that transport northward along the coast.

ψt = FieldTimeSeries(filenames[Ω], "ψ")
times = ψt.times

function gyre_transport(ψ, box)
    inside = Field{Face, Face, Nothing}(ψ.grid, Bool)
    set!(inside, (λ, φ) -> (box.longitude[1] < mod(λ, 360) < box.longitude[2]) &
                           (box.latitude[1] < φ < box.latitude[2]))
    return maximum(ψ; condition=inside) - minimum(ψ; condition=inside)
end

gulf_stream = (longitude = (275, 310), latitude = (20, 42))
kuroshio = (longitude = (118, 160), latitude = (20, 42))

# Now we compare the transport time series at the two rotation rates.

Sv = 1e6 # m³ s⁻¹

fig = Figure(size=(900, 400))
axes = (gulf_stream = Axis(fig[1, 1], title="Gulf Stream", xlabel="Time [years]", ylabel="Transport [Sv]"),
        kuroshio = Axis(fig[1, 2], title="Kuroshio", xlabel="Time [years]"))

colors = Dict(zip(rotation_rates, Makie.wong_colors()))

for rotation_rate in rotation_rates
    streamfunctions = FieldTimeSeries(filenames[rotation_rate], "ψ")
    label = coriolis_titles[rotation_rate]
    color = colors[rotation_rate]

    for (name, box) in ((:gulf_stream, gulf_stream), (:kuroshio, kuroshio))
        transport = [gyre_transport(streamfunctions[n], box) for n in 1:length(times)]
        lines!(axes[name], times / year, transport / Sv; label, color)
    end
end

axislegend(axes.gulf_stream, position=:lt)
save("western_boundary_current_transports.png", fig, px_per_unit=2) #hide

# ![](western_boundary_current_transports.png)
#
# Halving the rotation rate roughly doubles the transport of both boundary currents.
#
# ## The gyres
#
# Next we plot the streamfunction at the end of each simulation. The Antarctic
# Circumpolar Current puts a large offset between ``ψ`` on Antarctica and everywhere
# else, so we set ``ψ = 0`` on North America. Note the ten times larger color range of
# the ``f``-plane map.

height_cpu = on_architecture(CPU(), height)
land = interior(height_cpu, :, :, 1) .≥ 0

distance_to_north_america = Field{Center, Center, Nothing}(underlying_grid)
set!(distance_to_north_america, (λ, φ) -> (mod(λ, 360) - 260)^2 + (φ - 40)^2)
north_america = argmin(distance_to_north_america)

function streamfunction_map!(fig, row, filename; title, colorrange)
    streamfunctions = FieldTimeSeries(filename, "ψ")
    ψ_end = streamfunctions[end]
    reference = ψ_end[north_america]
    streamfunction = Field((ψ_end - reference) / Sv)
    ax = map_axis(fig[row, 1]; title)
    qm = quadmesh!(ax, streamfunction; colormap=:balance, colorrange, nan_color=:gray)
    Colorbar(fig[row, 2], qm, label="Streamfunction [Sv]")
    return nothing
end

experiments = ((filenames[Ω], coriolis_titles[Ω], (-100, 100)),
               (filenames[Ω / 2], coriolis_titles[Ω / 2], (-100, 100)),
               (f_plane_filename, "f = 2Ω sin 30°", (-1000, 1000)))

fig = Figure(size=(900, 1050))

for (row, (filename, title, colorrange)) in enumerate(experiments)
    streamfunction_map!(fig, row, filename; title, colorrange)
end

save("global_wind_driven_gyres.png", fig, px_per_unit=2) #hide

# ![](global_wind_driven_gyres.png)
#
# Halving the rotation rate doubles the gyres but leaves their shape alone: the
# streamfunction climbs to the gyre maximum within a few degrees of the western coast
# and decays slowly across the rest of the basin. On the ``f``-plane the gyres are
# symmetric about the middle of each basin and there is no western boundary current.
# They are also more than ten times stronger and take about two years to level off: without
# ``β`` there is no Sverdrup balance, so the wind keeps spinning up each basin until
# friction alone can remove the vorticity it puts in.
#
# ## Currents and temperature
#
# Finally we animate the global surface speed and, zoomed on the Gulf Stream and the
# Kuroshio, the departure of the surface temperature from its restoring profile,
# ``T - T^\star``, for the three Coriolis parameters. With ``f = 2Ω \sin φ`` the western
# boundary currents appear within the first weeks and then sharpen and speed up at the
# surface over the following years; with ``f = Ω \sin φ`` they are about 50% faster. On the
# ``f``-plane there are no boundary currents at all: the whole gyre circulates at a few
# tens of centimeters per second. The temperature spends its first two months relaxing
# from the uniform initial 10 °C toward ``T^\star``. After that, on the ``β``-planes, the
# boundary currents carry a tongue of water a degree or two warmer than ``T^\star``
# poleward along the coast, while the fast ``f``-plane gyres stir the whole basin into
# lobes several degrees warm and cold.

gulf_stream_view = (limits = ((260, 320), (15, 55)), xticks = (270:15:315, ["90°W", "75°W", "60°W", "45°W"]))
kuroshio_view = (limits = ((115, 175), (15, 55)), xticks = (120:15:165, ["120°E", "135°E", "150°E", "165°E"]))

tripolar_grid = TripolarGrid(CPU(), Float64; size=(Nx, Ny, 1), z=(0, 1)) ## the regridder needs Float64 coordinates
restoring_profile = Field{Center, Center, Nothing}(tripolar_grid)
set!(restoring_profile, (λ, φ) -> restoring_temperature(φ))

latitude_longitude_grid((longitude, latitude)) =
    LatitudeLongitudeGrid(CPU(), Float64; longitude, latitude, topology=(Bounded, Bounded, Flat),
                          size=(2 * (longitude[2] - longitude[1]), 2 * (latitude[2] - latitude[1])))

regions = ("Gulf Stream" => gulf_stream_view, "Kuroshio" => kuroshio_view)
regional_grids = [latitude_longitude_grid(zoom.limits) for (_, zoom) in regions]
regridders = [ConservativeRegridding.Regridder(grid, tripolar_grid) for grid in regional_grids]

function regrid_temperature_anomaly(temperatures)
    anomaly = Field{Center, Center, Nothing}(tripolar_grid)
    regional_anomalies = [FieldTimeSeries{Center, Center, Nothing}(grid, temperatures.times) for grid in regional_grids]

    for n in eachindex(temperatures.times)
        set!(anomaly, ifelse.(land, NaN, interior(temperatures[n], :, :, 1) .- interior(restoring_profile, :, :, 1)))
        for (anomalies, regridder) in zip(regional_anomalies, regridders)
            ConservativeRegridding.regrid!(anomalies[n], regridder, anomaly)
        end
    end

    return regional_anomalies
end

n = Observable(1)

fig = Figure(size=(1800, 1000))
Label(fig[1, 1:5], @lift(@sprintf("After %.1f years", times[$n] / year)); fontsize=22, tellwidth=false)

for (row, (filename, title, _)) in enumerate(experiments)
    speeds = FieldTimeSeries(filename, "surface_speed")
    speed = @lift speeds[$n]

    ax = map_axis(fig[row + 1, 1]; title="Surface speed, " * title)
    qm = quadmesh!(ax, speed; colormap=:magma, colorrange=(0, 0.3), nan_color=:gray)
    row == 1 && Colorbar(fig[2:4, 2], qm, label="Speed [m s⁻¹]")

    regional_anomalies = regrid_temperature_anomaly(FieldTimeSeries(filename, "surface_temperature"))

    for (column, ((region, zoom), anomalies)) in enumerate(zip(regions, regional_anomalies))
        ax = map_axis(fig[row + 1, column + 2]; title="T − T*, $region, " * title, zoom...)
        hm = heatmap!(ax, @lift(anomalies[$n]); colormap=:balance, colorrange=(-2, 2), nan_color=:gray)
        row == 1 && column == 2 && Colorbar(fig[2:4, 5], hm, label="T − T* [°C]")
    end
end

CairoMakie.record(fig, "global_wind_driven_gyres.mp4", 1:2:length(times), framerate=12) do frame
    n[] = frame
end
nothing #hide

# ![](global_wind_driven_gyres.mp4)
