# # Global wind-driven gyres on a tripolar grid
#
# This example is a stripped-down version of the global ocean configurations used for
# OMIP-style simulations: a [`TripolarGrid`](@ref) with realistic bathymetry,
# a ``z^\star`` vertical coordinate, and a [`SplitExplicitFreeSurface`](@ref).
# To keep it cheap enough for a laptop GPU, the grid is 1° with four layers.
#
# We force the ocean with an idealized zonal wind stress and look at the western boundary
# currents, the Gulf Stream and the Kuroshio, that close the wind-driven gyres.
# Sverdrup theory says that the depth-integrated meridional transport of the interior is
#
# ```math
# V = \frac{\boldsymbol{\hat z} \boldsymbol{\cdot} (\boldsymbol{\nabla} \times \boldsymbol{\tau})}{\rho₀ \beta} ,
# \qquad \beta = \frac{2 \Omega \cos \varphi}{R} ,
# ```
#
# and the western boundary current returns that transport back across the basin.
# Its strength should therefore scale with ``1 / \Omega``, which we check by running
# the simulation at Earth's rotation rate and at twice that. A third run with a constant
# Coriolis parameter shows that the gyres owe their western intensification to ``β``.
#
# ## Install dependencies
#
# Besides Oceananigans, this example needs NCDatasets to read the bathymetry and CairoMakie
# to make the plots. To run on a GPU you also need the Julia package for your GPU: CUDA for
# NVIDIA, Metal for Apple silicon, or AMDGPU for AMD. On a machine without a GPU, skip that
# package and the example runs on the CPU.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, CairoMakie"
# pkg"add CUDA" # or Metal, or AMDGPU
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: φnode
using Oceananigans.ImmersedBoundaries: InterfaceImmersedCondition
using NCDatasets
using Downloads
using Printf
using CairoMakie

# We run on the GPU if there is one: Metal on Apple silicon, CUDA if an NVIDIA driver is
# installed, AMDGPU if a ROCm driver is installed, and the CPU otherwise.
#
# To keep the demo fast, everything runs in single precision (`Float32`), which is much faster
# than double precision on most GPUs. For research-grade simulations we recommend `Float64`,
# which Metal does not currently support, so Apple GPUs are limited to `Float32`.

if Sys.isapple() && Sys.ARCH === :aarch64
    using Metal
    arch = Metal.functional() ? GPU(Metal.MetalBackend()) : CPU()
elseif !isnothing(Sys.which("nvidia-smi"))
    using CUDA
    arch = CUDA.functional() ? GPU() : CPU()
elseif !isnothing(Sys.which("rocminfo"))
    using AMDGPU
    arch = AMDGPU.functional() ? GPU(AMDGPU.ROCBackend()) : CPU()
else
    arch = CPU()
end

FT = Float32
Oceananigans.defaults.FloatType = FT

# ## A four-layer tripolar grid
#
# The tripolar grid spans the globe from 80°S to the North Pole. The resolution is a
# parameter: the three five-year 1° runs below take about an hour and a half on a laptop
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
# NOAA's ERDDAP server can subsample the 1-arc-minute ETOPO1 relief on the fly, so we
# download every `stride`-th point, three per grid cell, a file of a couple of megabytes.
# Averaging the samples in 3 × 3 blocks then gives the mean elevation of each grid cell
# rather than the elevation at a single point.

stride = round(Int, 20resolution) # arc-minutes between samples
etopo_url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude" *
            "[($(-90 + stride/120)):$stride:($(90 - stride/120))]" *
            "[($(-180 + stride/120)):$stride:($(180 - stride/120))]"

etopo_filename = "etopo1_$(stride)_arcmin.nc"
isfile(etopo_filename) || Downloads.download(etopo_url, etopo_filename)

etopo_elevation, etopo_longitude, etopo_latitude = NCDataset(etopo_filename) do dataset
    nomissing(dataset["altitude"][:, :]), dataset["longitude"][:], dataset["latitude"][:]
end

block_mean(a, n) = [sum(@view a[i:i+n-1, j:j+n-1]) / n^2 for i in 1:n:size(a, 1), j in 1:n:size(a, 2)]
elevation = FT.(block_mean(etopo_elevation, 3))

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

# The tripolar grid is curvilinear, so we draw maps with `surface!` on the grid's own
# longitudes and latitudes, which run from 70°E eastward around the globe, and hide the land.

λ = Array(λnodes(underlying_grid, Center(), Center(), Center()))
φ = Array(φnodes(underlying_grid, Center(), Center(), Center()))
depth = - Array(interior(bottom_height_field(grid), :, :, 1))
land = depth .≤ 0

longitude_ticks = (120:60:420, ["120°E", "180°", "120°W", "60°W", "0°", "60°E"])

map_axis(figure_position; title="") =
    Axis(figure_position; title, xlabel="Longitude", ylabel="Latitude",
         aspect=DataAspect(), limits=((70, 430), (-80, 70)), xticks=longitude_ticks)

fig = Figure(size=(900, 500))
ax = map_axis(fig[1, 1]; title="Ocean depth")
sf = surface!(ax, λ, φ, 0 * λ; color=ifelse.(land, NaN, depth), colormap=:deep,
              shading=NoShading, nan_color=:gray)
Colorbar(fig[1, 2], sf, label="Depth [m]")
save("bathymetry.png", fig, px_per_unit=2) #hide

# ![](bathymetry.png)

# ## Wind stress and bottom drag
#
# The zonal wind stress
#
# ```math
# τˣ(φ) = - τ₀ \, \sin 2φ \sin 6φ
# ```
#
# has easterly trade winds peaking at ±15°, westerlies peaking at ±45°, and polar
# easterlies peaking at ±75°. Its curl drives cyclonic tropical and subpolar gyres and
# anticyclonic subtropical gyres, whose western boundary currents are the Gulf Stream
# and the Kuroshio. A positive flux boundary condition transports momentum out of the
# domain, so the momentum flux from the wind is minus the wind stress divided by
# the reference density `ρ₀`.

τ₀ = FT(0.15)   # peak wind stress [N m⁻²]
ρ₀ = 1020       # reference density [kg m⁻³]

zonal_wind_stress(φ, τ₀) = - τ₀ * sind(2φ) * sind(6φ)
zonal_momentum_flux(λ, φ, t, parameters) = - zonal_wind_stress(φ, parameters.τ₀) / parameters.ρ₀

# We plot the wind stress together with the Sverdrup transport per unit zonal width
# that it drives at Earth's rotation rate. The wind stress curl is
# ``- R^{-1} \, ∂τˣ / ∂φ``.

R = Oceananigans.defaults.planet_radius
Ω = Oceananigans.defaults.planet_rotation_rate

wind_stress_curl(φ) = τ₀ / R * (2 * cosd(2φ) * sind(6φ) + 6 * sind(2φ) * cosd(6φ))
sverdrup_transport(φ, rotation_rate) = wind_stress_curl(φ) / (ρ₀ * 2 * rotation_rate * cosd(φ) / R)

latitudes = -80:0.5:80

fig = Figure(size=(800, 300))
ax = Axis(fig[1, 1], xlabel="Latitude [°]", ylabel="Zonal wind stress [N m⁻²]")
lines!(ax, latitudes, zonal_wind_stress.(latitudes, τ₀))
ax = Axis(fig[1, 2], xlabel="Latitude [°]", ylabel="Sverdrup transport [m² s⁻¹]")
lines!(ax, latitudes[abs.(latitudes) .> 5], sverdrup_transport.(latitudes[abs.(latitudes) .> 5], Ω))
save("wind_stress.png", fig, px_per_unit=2) #hide

# ![](wind_stress.png)

# The wind stress enters through the top boundary condition on `u`. A quadratic
# [`BulkDrag`](@ref) acts on the sea floor, which is the bottom of the domain in the
# deep ocean and an immersed boundary everywhere else.

wind_stress = FluxBoundaryCondition(zonal_momentum_flux, parameters=(; τ₀, ρ₀))
drag = BulkDrag(coefficient=FT(2.5e-3))
u_boundary_conditions = FieldBoundaryConditions(top=wind_stress, bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))
v_boundary_conditions = FieldBoundaryConditions(bottom=drag, immersed=ImmersedBoundaryCondition(bottom=drag))

# ## Surface temperature restoring
#
# The surface layer is restored on a 30-day time scale to ``T^\star(φ) = 30 \cos^2 φ``,
# warm at the equator and cold at the poles. The flux is written in discrete form so that
# it can read the surface temperature at each column.

restoring_temperature(φ) = 30 * cosd(φ)^2

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
# We use WENO advection schemes for momentum and for temperature, and no explicit
# viscosity or diffusivity apart from a convective adjustment that mixes statically
# unstable columns. Temperature sets the buoyancy through a linear equation of state.
# It starts from a horizontally uniform exponential thermocline with a 1 kilometer scale:
# a meridional density gradient across a basin comes with a depth-integrated thermal
# wind of hundreds of Sverdrups that would swamp the wind-driven gyres and take years
# to adjust away, so the equator-to-pole contrast enters only through the surface restoring.

momentum_advection = WENOVectorInvariant(order=5)
tracer_advection = WENO(order=7)
buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(thermal_expansion=2e-4), constant_salinity=35)
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
# the spherical ``f = 2Ω \sin φ``, at Earth's rotation rate and at twice it, which doubles
# ``β`` and should halve the Sverdrup transport. The third uses an [`FPlane`](@ref) with the
# value of ``f`` at 30°N everywhere, ``f = 2Ω \sin 30°``, so that ``β = 0``. A constant
# ``f`` has the wrong sign in the Southern Hemisphere, so on the ``f``-plane we only look
# at the northern gyres.
#
# The simulation runner saves the barotropic streamfunction ``ψ``, defined by
# ``U = ∫ u \, \mathrm{d} z = - ∂ψ / ∂y`` and computed by integrating ``U`` northward from
# Antarctica, together with the surface speed and the surface temperature, every ten days
# of a five-year run.

year = 360days

function run_gyres(grid, coriolis, name; stop_time=5year, save_interval=10days)
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

coriolis_title(rotation_rate) = "f = $(round(Int, 2rotation_rate / Ω))Ω sin φ"

rotation_rates = (Ω, 2Ω)
filenames = Dict(rotation_rate => run_gyres(grid, HydrostaticSphericalCoriolis(; rotation_rate), @sprintf("omega_%d", rotation_rate / Ω))
                 for rotation_rate in rotation_rates)

f_plane_filename = run_gyres(grid, FPlane(latitude=30), "f_plane")

# ## Gyre transports
#
# The transport of a gyre is the difference between the streamfunction at the gyre
# center and on the coast, so we measure it as the range of ``ψ`` over a box that
# contains the center of the subtropical gyre and the western boundary. The Gulf Stream
# and the Kuroshio carry that transport northward along the coast.

ψt = FieldTimeSeries(filenames[Ω], "ψ")
times = ψt.times

function gyre_transport(ψ, box)
    inside = @. (box.longitude[1] < mod(λ, 360) < box.longitude[2]) & (box.latitude[1] < φ < box.latitude[2])
    values = interior(ψ, :, :, 1)[inside]
    return maximum(values) - minimum(values)
end

gulf_stream = (longitude = (275, 310), latitude = (20, 42))
kuroshio = (longitude = (118, 160), latitude = (20, 42))

# For the Sverdrup prediction we integrate the interior transport across each basin
# on the ETOPO grid and take the largest value over the latitudes of the gyre.

etopo_ocean = etopo_elevation .< 0
etopo_longitude = mod.(etopo_longitude, 360)

function sverdrup_gyre_transport(rotation_rate, basin)
    transports = map(basin.latitude[1]:resolution:basin.latitude[2]) do latitude
        j = argmin(abs.(etopo_latitude .- latitude))
        inside = @. basin.longitude[1] < etopo_longitude < basin.longitude[2]
        width = count(etopo_ocean[:, j] .& inside) * 2π * R * cosd(latitude) / length(etopo_longitude)
        return - sverdrup_transport(latitude, rotation_rate) * width
    end
    return maximum(transports)
end

atlantic = (longitude = (280, 360), latitude = (20, 42))
pacific = (longitude = (120, 250), latitude = (20, 42))

# Now we compare the transport time series with the Sverdrup prediction, which
# halves when the rotation rate doubles.

Sv = 1e6 # m³ s⁻¹

fig = Figure(size=(900, 400))
axes = (gulf_stream = Axis(fig[1, 1], title="Gulf Stream", xlabel="Time [years]", ylabel="Transport [Sv]"),
        kuroshio = Axis(fig[1, 2], title="Kuroshio", xlabel="Time [years]"))

colors = Dict(zip(rotation_rates, Makie.wong_colors()))

for rotation_rate in rotation_rates
    streamfunctions = FieldTimeSeries(filenames[rotation_rate], "ψ")
    label = coriolis_title(rotation_rate)
    color = colors[rotation_rate]

    for (name, box, basin) in ((:gulf_stream, gulf_stream, atlantic), (:kuroshio, kuroshio, pacific))
        transport = [gyre_transport(streamfunctions[n], box) for n in 1:length(times)]
        lines!(axes[name], times / year, transport / Sv; label, color)
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
# Next we plot the streamfunction at the end of each simulation and follow it along
# 30°N across the Pacific and the Atlantic. The Antarctic Circumpolar Current puts a
# large offset between ``ψ`` on Antarctica and everywhere else, so we set ``ψ = 0`` on
# North America. Note the ten times larger color range of the ``f``-plane map.

north_america = argmin(@. (mod(λ, 360) - 260)^2 + (φ - 40)^2)

function streamfunction_map!(fig, row, filename; title, colorrange)
    streamfunctions = FieldTimeSeries(filename, "ψ")
    ψ_end = interior(streamfunctions[end], :, :, 1)
    streamfunction = ifelse.(land, NaN, (ψ_end .- ψ_end[north_america]) / Sv)
    axis = map_axis(fig[row, 1]; title)
    sf = surface!(axis, λ, φ, 0 * λ; color=streamfunction, colormap=:balance, colorrange,
                  shading=NoShading, nan_color=:gray)
    Colorbar(fig[row, 2], sf, label="Streamfunction [Sv]")
    return streamfunction
end

experiments = ((filenames[Ω], coriolis_title(Ω), (-100, 100)),
               (filenames[2Ω], coriolis_title(2Ω), (-100, 100)),
               (f_plane_filename, "f = 2Ω sin 30°", (-1000, 1000)))

along_30N = @. abs(φ - 30) < resolution / 2
eastward = sortperm(λ[along_30N])

fig = Figure(size=(900, 1400))
ax = Axis(fig[4, 1], xlabel="Longitude", ylabel="Streamfunction along 30°N [Sv]", xticks=longitude_ticks)

for (row, (filename, title, colorrange)) in enumerate(experiments)
    streamfunction = streamfunction_map!(fig, row, filename; title, colorrange)
    lines!(ax, λ[along_30N][eastward], streamfunction[along_30N][eastward]; label=title)
end

axislegend(ax)
save("global_wind_driven_gyres.png", fig, px_per_unit=2) #hide

# ![](global_wind_driven_gyres.png)
#
# Doubling the rotation rate halves the gyres but leaves their shape alone: the
# streamfunction climbs to the gyre maximum within a few degrees of the western coast
# and decays slowly across the rest of the basin. On the ``f``-plane the gyres are
# symmetric about the middle of each basin and there is no western boundary current.
# They are also twenty times stronger and take about a year to level off: without
# ``β`` there is no Sverdrup balance, so the wind keeps spinning up each basin until
# friction alone can remove the vorticity it puts in.
#
# ## Currents and temperature
#
# Finally we animate the surface speed and the departure of the surface temperature from
# its restoring profile, ``T - T^\star``, for the three Coriolis parameters. With
# ``f = 2Ω \sin φ`` the western boundary currents appear within the first weeks and then
# sharpen and speed up at the surface over the following years; with ``f = 4Ω \sin φ``
# they are half as fast. On the ``f``-plane there are no boundary currents at all: the
# whole gyre circulates at a few tens of centimeters per second. The temperature spends
# its first two months relaxing from the uniform initial 10 °C toward ``T^\star``. After
# that it stays within a few degrees of ``T^\star`` on the ``β``-planes, cooler along the
# equator where the Ekman divergence brings deeper water to the surface and warmer under
# the subtropical convergence and along the western boundaries, while the fast ``f``-plane
# gyres stir it into lobes several degrees warm and cold.

restoring_profile = restoring_temperature.(φ)

n = Observable(1)

fig = Figure(size=(1400, 1000))
Label(fig[1, 1:4], @lift(@sprintf("After %.1f years", times[$n] / year)); fontsize=22, tellwidth=false)

for (row, (filename, title, _)) in enumerate(experiments)
    speeds = FieldTimeSeries(filename, "surface_speed")
    temperatures = FieldTimeSeries(filename, "surface_temperature")
    speed = @lift ifelse.(land, NaN, interior(speeds[$n], :, :, 1))
    anomaly = @lift ifelse.(land, NaN, interior(temperatures[$n], :, :, 1) .- restoring_profile)

    ax = map_axis(fig[row + 1, 1]; title="Surface speed, " * title)
    sf = surface!(ax, λ, φ, 0 * λ; color=speed, colormap=:magma, colorrange=(0, 0.3),
                  shading=NoShading, nan_color=:gray)
    row == 1 && Colorbar(fig[2:4, 2], sf, label="Speed [m s⁻¹]")

    ax = map_axis(fig[row + 1, 3]; title="T − T*, " * title)
    sf = surface!(ax, λ, φ, 0 * λ; color=anomaly, colormap=:balance, colorrange=(-2, 2),
                  shading=NoShading, nan_color=:gray)
    row == 1 && Colorbar(fig[2:4, 4], sf, label="T − T* [°C]")
end

CairoMakie.record(fig, "global_wind_driven_gyres.mp4", 1:2:length(times), framerate=12) do frame
    n[] = frame
end
nothing #hide

# ![](global_wind_driven_gyres.mp4)
