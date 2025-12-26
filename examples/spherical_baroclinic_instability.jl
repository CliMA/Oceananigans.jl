# # Baroclinic instability on the sphere
#
# This example illustrates how to set up and run simulations of baroclinic instability
# on a spherical domain using three different spherical grid configurations:
# a standard latitude-longitude grid, a tripolar grid, and a rotated latitude-longitude grid.
#
# Baroclinic instability is a fundamental mechanism for generating mesoscale eddies in the
# ocean and synoptic-scale weather systems in the atmosphere. The instability arises when
# horizontal density gradients (fronts) are tilted by the combined effects of Earth's rotation
# and stratification, converting available potential energy into kinetic energy. This process
# is described in detail by Vallis (2017) and the classic paper by Eady (1949) provides
# foundational theory.
#
# In this example, we initialize a meridional temperature front that is baroclinically unstable,
# and watch eddies grow and equilibrate the front. We demonstrate this phenomenon on three
# different spherical grid types to illustrate the flexibility of Oceananigans for global
# ocean modeling.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie, SeawaterPolynomials"
# ```

# ## Grid configurations
#
# We set up three different spherical grids at 4-degree resolution. Each grid type has
# different advantages:
#
# - **LatitudeLongitudeGrid**: The most intuitive, but suffers from converging meridians at
#   the poles, which requires filtering or limiting latitudinal extent.
#
# - **TripolarGrid**: Avoids the North Pole singularity by introducing two computational poles
#   over land (typically over North America and Eurasia). This grid was first introduced by
#   Murray (1996) and is widely used in global ocean models.
#
# - **RotatedLatitudeLongitudeGrid**: Rotates the grid's north pole to an arbitrary location,
#   allowing finer resolution in a region of interest while avoiding the geographic poles.

using Oceananigans
using Oceananigans.Units
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using CUDA
using Printf

# Set up resolution and grid parameters. We use 4-degree resolution for
# reasonable runtimes while still resolving the instability.

arch = GPU()
resolution = 1             # degrees
Nx = 360 ÷ resolution      # number of longitude points
Ny = 170 ÷ resolution      # number of latitude points (avoiding poles)
Nz = 10                    # number of vertical levels
size = (Nx, Ny, Nz)
halo = (7, 7, 7)           # halo size for higher-order advection schemes
H = 3000                   # domain depth [m]
latitude = (-85, 85)       # latitude range (avoiding poles for lat-lon grid)
longitude = (0, 360)       # longitude range
z = (-H, 0)                # vertical extent

# Build the three grids

lat_lon_grid = LatitudeLongitudeGrid(arch; size, halo, latitude, longitude, z)

# For the tripolar grid, we set up singularities ("north poles") at 55°N latitude.
# We also use an `ImmersedBoundaryGrid` to place cylindrical islands over the singularities
# to ensure the simulation remains stable.

underlying_tripolar_grid = TripolarGrid(arch; size, halo, z)

# Create cylindrical islands over the North Pole singularities to mask them out.
# The tripolar grid places singularities at longitude `first_pole_longitude` and
# `first_pole_longitude + 180°`, both at latitude `north_poles_latitude`.

dφ, dλ = 4, 8     # island extent in latitude and longitude
λ₀, φ₀ = 70, 55   # first pole location
h = 100           # island height above the bottom

cylinder(λ, φ) = ((λ - λ₀)^2 / 2dλ^2 + (φ - φ₀)^2 / 2dφ^2) < 1
cylindrical_isles(λ, φ) = -H + (H + h) * (cylinder(λ, φ) + cylinder(λ - 180, φ))

tripolar_grid = ImmersedBoundaryGrid(underlying_tripolar_grid, GridFittedBottom(cylindrical_isles))

# The rotated latitude-longitude grid rotates the north pole to an arbitrary location.
# Here we place the grid's north pole at (70°E, 55°N).

rotated_lat_lon_grid = RotatedLatitudeLongitudeGrid(arch; size, halo, latitude, longitude, z,
                                                    north_pole = (70, 55))

# ## Model setup
#
# We create a function that builds a `HydrostaticFreeSurfaceModel` for any of our grids.
# The model uses:
# - WENO advection for both momentum and tracers
# - Spherical Coriolis force appropriate for hydrostatic dynamics
# - Realistic seawater buoyancy using the TEOS-10 equation of state
# - Split-explicit free surface for fast external gravity wave dynamics

function build_model(grid)
    momentum_advection = WENOVectorInvariant(order=9)
    tracer_advection = WENO(order=7)
    coriolis = HydrostaticSphericalCoriolis()
    equation_of_state = TEOS10EquationOfState()
    buoyancy = SeawaterBuoyancy(; equation_of_state)
    free_surface = SplitExplicitFreeSurface(grid; substeps=120)
    model = HydrostaticFreeSurfaceModel(; grid, coriolis, free_surface, buoyancy, tracers = (:T, :S),
                                        momentum_advection, tracer_advection)
    return model
end

# ## Initial conditions
#
# We initialize with a temperature front in the meridional direction:
# - Warm water in the tropics, cold water at high latitudes
# - The front is centered around ±45° latitude
# - Random noise seeds the baroclinic instability
#
# Salinity is initialized with a linear vertical profile plus random perturbations.

Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()

# ## Run a simulation
#
# We define a function that sets up and runs a simulation on a given grid.
# We run for 30 days to observe the initial development of the instability
# while keeping computational costs reasonable.

function run_baroclinic_instability(grid, name; stop_time=30days, save_interval=12hours)
    model = build_model(grid)
    set!(model, T=Tᵢ, S=Sᵢ)

    simulation = Simulation(model; Δt=5minutes, stop_time)

    # Add progress callback
    function progress(sim)
        T = sim.model.tracers.T
        u, v, w = sim.model.velocities

        msg = @sprintf("%s, iteration %d: %s, max|u|: (%.2e, %.2e, %.2e)",
                       name, iteration(sim), prettytime(sim),
                       maximum(abs, u), maximum(abs, v), maximum(abs, w))

        msg *= @sprintf(", T ∈ (%.2f, %.2f)", minimum(T), maximum(T))

        @info msg

        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(1000))

    # Set up output: save vorticity and temperature at the surface
    u, v, w = model.velocities
    T = model.tracers.T
    ζ = ∂x(v) - ∂y(u)
    ∇T = sqrt(∂x(T)^2 + ∂y(T)^2)

    filename = "spherical_baroclinic_instability_" * name
    indices = (:, :, grid.Nz)
    fields = (; ζ, T, ∇T)

    simulation.output_writers[:surface] = JLD2Writer(model, fields; indices,
                                                     filename = filename * ".jld2",
                                                     schedule = TimeInterval(save_interval),
                                                     overwrite_existing = true)
    @info "Running $name simulation..."
    run!(simulation)

    @info "$name simulation completed in $(prettytime(simulation.run_wall_time))."

    return simulation
end

# ## Run the simulations
#
# Now we run simulations on all three grids. We use a shorter runtime
# to demonstrate the code.

simulations = Dict(
    "lat_lon" => run_baroclinic_instability(lat_lon_grid, "lat_lon"),
    "tripolar" => run_baroclinic_instability(tripolar_grid, "tripolar"),
    "rotated_lat_lon" => run_baroclinic_instability(rotated_lat_lon_grid, "rotated_lat_lon")
)

# ## Visualization
#
# We visualize the results using CairoMakie. First we make a 2D comparison
# of the three grid types, then we demonstrate 3D globe visualization using
# the `geo_surface!` utility from the Oceananigans Makie extension.

using CairoMakie

# Load the output from each simulation

grid_names = ["lat_lon", "tripolar", "rotated_lat_lon"]
grid_labels = ["Latitude-Longitude", "Tripolar", "Rotated Lat-Lon"]

T_ts = Dict()
ζ_ts = Dict()

for name in grid_names
    filename = "spherical_baroclinic_instability_" * name * ".jld2"
    T_ts[name] = FieldTimeSeries(filename, "T")
    ζ_ts[name] = FieldTimeSeries(filename, "ζ")
end

times = T_ts["lat_lon"].times
Nt = length(times)

# ### 2D comparison of all three grids
#
# Create a comparison figure showing the final state for each grid

fig = Figure(size = (1600, 600))

for (i, (name, label)) in enumerate(zip(grid_names, grid_labels))
    Tn = interior(T_ts[name][Nt], :, :, 1)
    ζn = interior(ζ_ts[name][Nt], :, :, 1)

    ax_T = Axis(fig[1, i]; title = "$label: Temperature", xlabel = "i", ylabel = "j")
    ax_ζ = Axis(fig[2, i]; title = "$label: Vorticity", xlabel = "i", ylabel = "j")

    heatmap!(ax_T, Tn; colormap = :thermal, colorrange = (5, 30))
    heatmap!(ax_ζ, ζn; colormap = :balance, colorrange = (-5e-5, 5e-5))
end

save("spherical_baroclinic_instability_comparison.png", fig)
nothing #hide

# ![](spherical_baroclinic_instability_comparison.png)

# ### 3D globe visualization
#
# The Oceananigans Makie extension automatically handles spherical grids when
# using `surface!` on an `Axis3`. Simply pass the field directly and it will
# be plotted on a 3D sphere.

fig = Figure(size = (1000, 800))

ax3d = Axis3(fig[1, 1];
             aspect = :data,
             title = "Surface Temperature on a 3D Globe",
             xlabel = "x", ylabel = "y", zlabel = "z")

# Get the final temperature field
T_final = T_ts["lat_lon"][Nt]

# Plot the temperature on the sphere - the Makie extension handles the
# coordinate transformation automatically!
surface!(ax3d, T_final; colormap = :thermal, colorrange = (5, 30))

save("spherical_baroclinic_instability_globe.png", fig)
nothing #hide

# ![](spherical_baroclinic_instability_globe.png)

# ### Animation of the instability evolution
#
# Next we create a movie showing the evolution of the baroclinic instability
# on all three grids, visualized on 3D spheres. Each column shows a different
# grid type, with temperature on top and vorticity on the bottom.

fig = Figure(size = (1800, 1000))

n = Observable(1)

title = @lift "Baroclinic instability at t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 28)

# Create axes for each grid type (columns) and field (rows)
axes_T = Dict{String, Axis3}()
axes_ζ = Dict{String, Axis3}()

for (col, (name, label)) in enumerate(zip(grid_names, grid_labels))
    axes_T[name] = Axis3(fig[1, col]; aspect = :data, title = "$label\nTemperature",
                         xlabel = "", ylabel = "", zlabel = "")
    axes_ζ[name] = Axis3(fig[2, col]; aspect = :data, title = "Vorticity",
                         xlabel = "", ylabel = "", zlabel = "")
end

# Create the surface plots using Observable fields
plots_T = Dict{String, Any}()
plots_ζ = Dict{String, Any}()

for name in grid_names
    Tn = @lift T_ts[name][$n]
    ζn = @lift ζ_ts[name][$n]
    plots_T[name] = surface!(axes_T[name], Tn; colormap = :thermal, colorrange = (5, 30))
    plots_ζ[name] = surface!(axes_ζ[name], ζn; colormap = :balance, colorrange = (-5e-5, 5e-5))
end

# Add colorbars
Colorbar(fig[1, 4], plots_T["lat_lon"]; label = "Temperature [°C]")
Colorbar(fig[2, 4], plots_ζ["lat_lon"]; label = "Vorticity [s⁻¹]")

frames = 1:Nt
record(fig, "spherical_baroclinic_instability.mp4", frames; framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](spherical_baroclinic_instability.mp4)

# ## References
#
# - Eady, E. T. (1949). Long waves and cyclone waves. *Tellus*, 1(3), 33-52.
#   doi:[10.1111/j.2153-3490.1949.tb01265.x](https://doi.org/10.1111/j.2153-3490.1949.tb01265.x)
#
# - Murray, R. J. (1996). Explicit generation of orthogonal grids for ocean models.
#   *Journal of Computational Physics*, 126(2), 251-273.
#   doi:[10.1006/jcph.1996.0136](https://doi.org/10.1006/jcph.1996.0136)
#
# - Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics: Fundamentals and
#   Large-Scale Circulation* (2nd ed.). Cambridge University Press.
#   doi:[10.1017/9781107588417](https://doi.org/10.1017/9781107588417)

