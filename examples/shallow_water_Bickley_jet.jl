# # An unstable Bickley jet in Shallow Water model 
#
# This example uses Oceananigans.jl's `ShallowWaterModel` to simulate
# the evolution of an unstable, geostrophically balanced, Bickley jet
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
using Oceananigans.Models: ShallowWaterModel

# ## Two-dimensional domain 
#
# The shallow water model is a two-dimensional model and thus the number of vertical
# points `Nz` must be set to one.  Note that ``L_z`` is the mean depth of the fluid. 

Lx, Ly, Lz = 2π, 20, 10
Nx, Ny = 128, 128

grid = RectilinearGrid(size = (Nx, Ny),
                       x = (0, Lx), y = (-Ly/2, Ly/2),
                       topology = (Periodic, Bounded, Flat))

# ## Building a `ShallowWaterModel`
#
# We build a `ShallowWaterModel` with the `WENO5` advection scheme,
# 3rd-order Runge-Kutta time-stepping, non-dimensional Coriolis and
# gravitational acceleration

gravitational_acceleration = 1
coriolis = FPlane(f=1)

model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          timestepper = :RungeKutta3,
                          advection = WENO5())

# Use `architecture = GPU()` to run this problem on a GPU.

# ## Background state and perturbation
#
# The background velocity ``ū`` and free-surface ``η̄`` correspond to a
# geostrophically balanced Bickely jet with maximum speed of ``U`` and maximum 
# free-surface deformation of ``Δη``,

U = 1 # Maximum jet velocity
f = coriolis.f
g = gravitational_acceleration
Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

h̄(x, y, z) = Lz - Δη * tanh(y)
ū(x, y, z) = U * sech(y)^2

# The total height of the fluid is ``h = L_z + \eta``. Linear stability theory predicts that 
# for the parameters we consider here, the growth rate for the most unstable mode that fits 
# our domain is approximately ``0.139``.

# The vorticity of the background state is

ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)

# The initial conditions include a small-amplitude perturbation that decays away from the 
# center of the jet.

small_amplitude = 1e-4
 
 uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * randn()
uhⁱ(x, y, z) = uⁱ(x, y, z) * h̄(x, y, z)

# We first set a "clean" initial condition without noise for the purpose of discretely
# calculating the initial 'mean' vorticity,

ū̄h(x, y, z) = ū(x, y, z) * h̄(x, y, z)

set!(model, uh = ū̄h, h = h̄)

# We next compute the initial vorticity and perturbation vorticity,

uh, vh, h = model.solution

## Build velocities
u = uh / h
v = vh / h

## Build and compute mean vorticity discretely
ω = Field(∂x(v) - ∂y(u))
compute!(ω)

## Copy mean vorticity to a new field
ωⁱ = Field{Face, Face, Nothing}(model.grid)
ωⁱ .= ω

## Use this new field to compute the perturbation vorticity
ω′ = Field(ω - ωⁱ)

# and finally set the "true" initial condition with noise,

set!(model, uh = uhⁱ)

# ## Running a `Simulation`
#
# We pick the time-step so that we make sure we resolve the surface gravity waves, which 
# propagate with speed of the order ``\sqrt{g L_z}``. That is, with `Δt = 1e-2` we ensure 
# that `` \sqrt{g L_z} Δt / Δx,  \sqrt{g L_z} Δt / Δy < 0.7``.

simulation = Simulation(model, Δt = 1e-2, stop_time = 150)

# ## Prepare output files
#
# Define a function to compute the norm of the perturbation on the cross channel velocity.
# We obtain the `norm` function from `LinearAlgebra`.

using LinearAlgebra: norm

perturbation_norm(args...) = norm(v)

# Build the `output_writer` for the two-dimensional fields to be output.
# Output every `t = 1.0`.

fields_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_fields.nc")
simulation.output_writers[:fields] = NetCDFOutputWriter(model, (; ω, ω′),
                                                        filename = fields_filename,
                                                        schedule = TimeInterval(1),
                                                        overwrite_existing = true)

# Build the `output_writer` for the growth rate, which is a scalar field.
# Output every time step.

growth_filename = joinpath(@__DIR__, "shallow_water_Bickley_jet_perturbation_norm.nc")
simulation.output_writers[:growth] = NetCDFOutputWriter(model, (; perturbation_norm),
                                                        filename = growth_filename,
                                                        schedule = IterationInterval(1),
                                                        dimensions = (; perturbation_norm = ()),
                                                        overwrite_existing = true)

# And finally run the simulation.

run!(simulation)

# ## Visualize the results

# Load required packages to read output and plot.

using NCDatasets, Printf, CairoMakie
nothing # hide

# Define the coordinates for plotting.

x, y = xnodes(ω), ynodes(ω)
nothing # hide

# Read in the `output_writer` for the two-dimensional fields and then create an animation 
# showing both the total and perturbation vorticities.

fig = Figure(resolution = (800, 440))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = AxisAspect(1),
               limits = ((0, Lx), (-Ly/2, Ly/2)))

ax_ω  = Axis(fig[2, 1]; title = "Total vorticity, ω", axis_kwargs...)
ax_ω′ = Axis(fig[2, 3]; title = "Perturbation vorticity, ω - ω̄", axis_kwargs...)

n = Observable(1)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

times = ds["time"][:]

ω = @lift ds["ω"][:, :, 1, $n]
hm_ω = heatmap!(ax_ω, x, y, ω, colorrange = (-1, 1), colormap = :balance)
Colorbar(fig[2, 2], hm_ω)

ω′ = @lift ds["ω′"][:, :, 1, $n]
hm_ω′ = heatmap!(ax_ω′, x, y, ω′, colormap = :balance)
Colorbar(fig[2, 4], hm_ω′)

title = @lift @sprintf("t = %.1f", times[$n])
fig[1, 1:4] = Label(fig, title, textsize=24, tellwidth=false)

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "shallow_water_Bickley_jet.mp4", frames, framerate=12) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")
    n[] = i
end
nothing #hide

# ![](shallow_water_Bickley_jet.mp4)

# It's always good practice to close the NetCDF files when we are done.

close(ds)

# Read in the `output_writer` for the scalar field (the norm of ``v``-velocity).

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

     t = ds2["time"][:]
norm_v = ds2["perturbation_norm"][:]

close(ds2)
nothing # hide

# We import the `fit` function from `Polynomials.jl` to compute the best-fit slope of the 
# perturbation norm on a logarithmic plot. This slope corresponds to the growth rate.

using Polynomials: fit

I = 6000:7000

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

current_figure() # hide

# The slope of the best-fit curve on a logarithmic scale approximates the rate at which instability
# grows in the simulation. Let's see how this compares with the theoretical growth rate.

println("Numerical growth rate is approximated to be ", round(slope, digits=3), ",\n",
        "which is very close to the theoretical value of 0.139.")
