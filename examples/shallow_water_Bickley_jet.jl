# # An unstable Bickley jet in a shallow water model
#
# This example uses Oceananigans.jl's `ShallowWaterModel` to simulate
# the evolution of an unstable, geostrophically balanced Bickley jet.
# The example is periodic in ``x`` with flat bathymetry and
# uses the conservative formulation of the shallow water equations.
# The initial conditions superpose the Bickley jet with small-amplitude perturbations.
# See ["The nonlinear evolution of barotropically unstable jets," J. Phys. Oceanogr. (2003)](https://doi.org/10.1175/1520-0485(2003)033<2173:TNEOBU>2.0.CO;2)
# for more details on this problem.
#
# The mass transport ``(uh, vh)`` is the prognostic momentum variable
# in the conservative formulation of the shallow water equations,
# where ``(u, v)`` are the horizontal velocity components and ``h``
# is the layer height.
#
# ## Install dependencies
#
# First we make sure that we have all of the packages that are required to
# run the simulation.
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, Polynomials, CairoMakie"
# ```

using Oceananigans
using Random

Random.seed!(90210) # for reproducible results

# ## Two-dimensional domain
#
# The shallow water model is two-dimensional and uses grids that are `Flat`
# in the vertical direction. We use length scales non-dimensionalized by the width
# of the Bickley jet.

grid = RectilinearGrid(size = (48, 128),
                       x = (0, 2π),
                       y = (-10, 10),
                       topology = (Periodic, Bounded, Flat))

# ## Building a `ShallowWaterModel`
#
# We build a `ShallowWaterModel` with the `WENO` advection scheme,
# 3rd-order Runge-Kutta time-stepping, non-dimensional Coriolis, and
# gravitational acceleration,

gravitational_acceleration = 1
coriolis = FPlane(f=1)

model = ShallowWaterModel(grid; coriolis, gravitational_acceleration,
                          timestepper = :RungeKutta3,
                          momentum_advection = WENO())

# ## Background state and perturbation
#
# The background velocity ``ū`` and layer height ``h̄`` correspond to a
# geostrophically balanced Bickley jet with maximum speed of ``U`` and maximum
# free-surface deformation of ``Δη``,

U = 1  # maximum jet velocity
H = 10 # reference depth
f = coriolis.f
g = gravitational_acceleration
Δη = f * U / g

h̄(x, y) = H - Δη * tanh(y)
ū(x, y) = U * sech(y)^2

# The total height of the fluid is ``h = H + η``, where ``η`` is the free-surface displacement.
# Linear stability theory predicts that for the parameters we consider here, the growth rate
# for the most unstable mode that fits our domain is approximately ``0.139``.
#
# The initial conditions include a small-amplitude perturbation that decays away from the
# center of the jet.

small_amplitude = 1e-4

 uⁱ(x, y) = ū(x, y) + small_amplitude * exp(-y^2) * randn()
uhⁱ(x, y) = uⁱ(x, y) * h̄(x, y)

# We first set a "clean" initial condition without noise for the purpose of discretely
# calculating the background vorticity ``ω̄``,

ūh̄(x, y) = ū(x, y) * h̄(x, y)

set!(model, uh = ūh̄, h = h̄)

# We next compute the vorticity ``ω = ∂_x v - ∂_y u``, store a copy of its initial value
# as ``ω̄``, and define the perturbation vorticity ``ω′ = ω - ω̄``,

uh, vh, h = model.solution

u = uh / h
v = vh / h

ω = Field(∂x(v) - ∂y(u))

ω̄ = Field{Face, Face, Nothing}(grid)
set!(ω̄, ω)

ω′ = Field(ω - ω̄)

# and finally set the "true" initial condition with noise,

set!(model, uh = uhⁱ)

# ## Running a `Simulation`
#
# We pick the time-step so that we make sure we resolve the surface gravity waves, which
# propagate with speed of the order ``\sqrt{g H}``. That is, with `Δt = 1e-2` we ensure
# that ``\sqrt{g H} Δt / Δx, \sqrt{g H} Δt / Δy < 0.7``.

simulation = Simulation(model, Δt = 1e-2, stop_time = 100)

# ## Prepare output files
#
# Define a function to compute the norm of the cross-channel velocity ``v``, which
# measures the perturbation amplitude. We obtain the `norm` function from `LinearAlgebra`.

using LinearAlgebra: norm

perturbation_norm(args...) = norm(v)

# Build the output writer for the two-dimensional vorticity fields, which outputs
# every 2 time units. Note that we need `NCDatasets` to be able to use the `NetCDFWriter`.

using NCDatasets

fields_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc")
simulation.output_writers[:fields] = NetCDFWriter(model, (; ω, ω′),
                                                  filename = fields_filename,
                                                  schedule = TimeInterval(2),
                                                  overwrite_files = true)

# Build the output writer for the perturbation norm, which is a scalar,
# and output it every time step.

growth_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc")
simulation.output_writers[:growth] = NetCDFWriter(model, (; perturbation_norm),
                                                  filename = growth_filename,
                                                  schedule = IterationInterval(1),
                                                  dimensions = (; perturbation_norm = ()),
                                                  overwrite_files = true)

# And finally run the simulation.

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Visualize the results
#
# We load the vorticity output as `FieldTimeSeries` and then create an animation
# showing both the total and perturbation vorticities.

using CairoMakie

ωt  = FieldTimeSeries(fields_filename, "ω")
ω′t = FieldTimeSeries(fields_filename, "ω′")

times = ωt.times

fig = Figure(size = (1200, 660))

axis_kwargs = (xlabel = "x", ylabel = "y")
ax_ω  = Axis(fig[2, 1]; title = "Total vorticity, ω", axis_kwargs...)
ax_ω′ = Axis(fig[2, 3]; title = "Perturbation vorticity, ω - ω̄", axis_kwargs...)

n = Observable(1)

ωn  = @lift ωt[$n]
ω′n = @lift ω′t[$n]

hm_ω = heatmap!(ax_ω, ωn, colorrange = (-1, 1), colormap = :balance)
Colorbar(fig[2, 2], hm_ω)

hm_ω′ = heatmap!(ax_ω′, ω′n, colormap = :balance)
Colorbar(fig[2, 4], hm_ω′)

title = @lift "t = " * string(round(times[$n], digits=1))
fig[1, 1:4] = Label(fig, title, fontsize=24, tellwidth=false)

current_figure() #hide
fig

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "shallow_water_Bickley_jet.mp4", frames, framerate=12) do i
    n[] = i
end
nothing #hide

# ![](shallow_water_Bickley_jet.mp4)

# Next, we read the time series of the perturbation norm, closing the NetCDF file
# when we are done.

ds = NCDataset(growth_filename)

t = ds["time"][:]
norm_v = ds["perturbation_norm"][:]

close(ds)
nothing #hide

# We import the `fit` function from `Polynomials.jl` to compute the best-fit slope of the
# perturbation norm on a logarithmic plot. This slope corresponds to the growth rate.

using Polynomials: fit

I = 5000:6000

degree = 1
linear_fit_polynomial = fit(t[I], log.(norm_v[I]), degree, var = :t)

# We can get the coefficient of the ``n``-th power from the fitted polynomial by using `n`
# as an index, e.g.,

constant, slope = linear_fit_polynomial[0], linear_fit_polynomial[1]

# We then use the computed linear fit coefficients to construct the best fit and plot it
# together with the time-series for the perturbation norm for comparison.

best_fit = @. exp(constant + slope * t)

lines(t, norm_v;
      linewidth = 4,
      label = "norm(v)",
      axis = (yscale = log10,
              limits = (nothing, (1e-3, 30)),
              xlabel = "time",
              ylabel = "norm(v)",
               title = "growth of perturbation norm"))

lines!(t[I], 2 * best_fit[I]; # factor 2 offsets fit from curve for better visualization
       linewidth = 4,
       label = "best fit")

axislegend(position = :rb)

current_figure() #hide

# The slope of the best-fit curve on a logarithmic scale approximates the rate at which instability
# grows in the simulation. Let's see how this compares with the theoretical growth rate.

println("Numerical growth rate is approximated to be ", round(slope, digits=3), ",\n",
        "which is very close to the theoretical value of 0.139.")
