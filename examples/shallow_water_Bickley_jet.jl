# # An unstable Bickley jet in Shallow Water model 
#
# This example shows how to use `Oceananigans.ShallowWaterModel` to simulate
# the evolution of an unstable, geostrophically balanced, Bickley jet.
# The model solves the governing equations for the shallow water model in
# conservative form.  The geometry is that of a periodic channel
# in the ``x``-direction with a flat bottom and a free-surface. The initial
# conditions are that of a Bickley jet with small-amplitude perturbations.
# The interested reader can see ["The nonlinear evolution of barotropically unstable jets," J. Phys. Oceanogr. (2003)](https://doi.org/10.1175/1520-0485(2003)033<2173:TNEOBU>2.0.CO;2)
# for more details on this specific problem. 
#
# Unlike the other models, the fields that are simulated are the mass transports, 
# ``uh`` and ``vh`` in the ``x`` and ``y`` directions, respectively,
# and the height ``h``.  Note that the velocities ``u`` and ``v`` are not state 
# variables but can be easily computed when needed, e.g., via `u = uh / h`.
#
# ## Install dependencies
#
# First we make sure that we have all of the packages that are required to
# run the simulation.
# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, Plots, Printf, Polynomials"
# ```

using Oceananigans
using Oceananigans.Models: ShallowWaterModel

# ## Two-dimensional domain 
#
# The shallow water model is a two-dimensional model and thus the number of vertical
# points `Nz` must be set to one.  Note that ``L_z`` is the mean depth of the fluid. 

Lx, Ly, Lz = 2π, 20, 1
Nx, Ny = 128, 128

grid = RegularRectilinearGrid(size = (Nx, Ny),
                              x = (0, Lx), y = (-Ly/2, Ly/2),
                              topology = (Periodic, Bounded, Flat))

# ## Physical parameters
#
# This is a toy problem and we choose the parameters so the jet idealizes a relatively narrow 
# mesoscale jet. The physical parameters are
#
#   * ``f``: Coriolis parameter
#   * ``g``: Acceleration due to gravity
#   * ``U``: Maximum jet speed
#   * ``\Delta \eta``: Maximum free-surface deformation as dictated by geostrophy

 const f = 1
 const g = 9.8
 const U = 1.0
const Δη = f * U / g
nothing # hide

# ## Building a `ShallowWaterModel`
#
# We use `grid`, `coriolis` and `gravitational_acceleration` to build the model.
# Furthermore, we specify `RungeKutta3` for time-stepping and `WENO5` for advection.

model = ShallowWaterModel(
    timestepper=:RungeKutta3,
    advection=WENO5(),
    grid=grid,
    gravitational_acceleration=g,
    coriolis=FPlane(f=f),
    )

# ## Background state and perturbation
#
# The background velocity ``ū`` and free-surface ``η̄`` are chosen to represent a 
# geostrophically balanced Bickely jet with maximum speed of ``U`` and maximum 
# free-surface deformation of ``Δη``, i.e.,
#
# ```math
# \begin{align}
# η̄(y) & = - Δη \tanh(y) , \\
# ū(y) & = U \mathrm{sech}^2(y) .
# \end{align}
# ```
#
# The total height of the fluid is ``h = L_z + \eta``. Linear stability theory predicts that 
# for the parameters we consider here, the growth rate for the most unstable mode that fits 
# our domain is approximately ``0.139``.
# 
# We also specify `ω̄` as the vorticity of the background state, 
# ``ω̄ = - ∂_y ū = 2 U \mathrm{sech}^2(y) \tanh(y)``.

h̄(x, y, z) = Lz - Δη * tanh(y)
ū(x, y, z) = U * sech(y)^2
ω̄(x, y, z) = 2 * U * sech(y)^2 * tanh(y)
nothing # hide

# The initial conditions include a small-amplitude perturbation that decays away from the 
# center of the jet.

 small_amplitude = 1e-4
 
 uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * randn()
 hⁱ(x, y, z) = h̄(x, y, z)
uhⁱ(x, y, z) = uⁱ(x, y, z) * hⁱ(x, y, z)
nothing # hide

# We set the initial conditions for the zonal mass transport `uhⁱ` and the fluid height `hⁱ`.

set!(model, uh = uhⁱ, h = hⁱ)

# We compute the total vorticity and the perturbation vorticity.

uh, vh, h = model.solution

        u = ComputedField(uh / h)
        v = ComputedField(vh / h)
        ω = ComputedField(∂x(v) - ∂y(u))
   ω_pert = ComputedField(ω - ω̄)
nothing #hide

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

function perturbation_norm(model)
    compute!(v)
    return norm(v)
end
nothing # hide

# Choose the two fields to be output to be the total and perturbation vorticity.

outputs = (ω_total = ω, ω_pert = ω_pert)

# Build the `output_writer` for the two-dimensional fields to be output.
# Output every `t = 1.0`.

simulation.output_writers[:fields] =
    NetCDFOutputWriter(
        model,
        (ω = ω, ω_pert = ω_pert),
          filepath = joinpath(@__DIR__, "shallow_water_Bickley_jet.nc"),
          schedule = TimeInterval(1.0),
              mode = "c")

# Build the `output_writer` for the growth rate, which is a scalar field.
# Output every time step.

simulation.output_writers[:growth] =
    NetCDFOutputWriter(
        model,
        (perturbation_norm = perturbation_norm,),
          filepath = joinpath(@__DIR__, "perturbation_norm_shallow_water.nc"),
          schedule = IterationInterval(1),
        dimensions = (perturbation_norm=(),),
              mode = "c")

# And finally run the simulation.

run!(simulation)

# ## Visualize the results

# Load required packages to read output and plot.

using NCDatasets, Plots, Printf
nothing # hide

# Define the coordinates for plotting.

x, y = xnodes(ω), ynodes(ω)
nothing # hide

# Define keyword arguments for plotting the contours.

kwargs = (
         xlabel = "x",
         ylabel = "y",
         aspect = 1,
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           ylim = (-Ly/2, Ly/2),
           xlim = (0, Lx)
)
nothing # hide

# Read in the `output_writer` for the two-dimensional fields and then create an animation 
# showing both the total and perturbation vorticities.

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

anim = @animate for (iter, t) in enumerate(ds["time"])
     ω = ds["ω"][:, :, 1, iter]
    ωp = ds["ω_pert"][:, :, 1, iter]

    ωp_max = maximum(abs, ωp)

     plot_ω = contour(x, y, ω',
                       clim = (-1, 1), 
                      title = @sprintf("Total vorticity, ω, at t = %.1f", t); kwargs...)
                      
    plot_ωp = contour(x, y, ωp',
                       clim = (-ωp_max, ωp_max),
                      title = @sprintf("Perturbation vorticity, ω - ω̄, at t = %.1f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1, 2), size = (800, 440))
end

close(ds)

mp4(anim, "Bickley_Jet_ShallowWater.mp4", fps=15)

# Read in the `output_writer` for the scalar field (the norm of ``v``-velocity).

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

iterations = keys(ds2["time"])

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

plot(t, norm_v,
        yaxis = :log,
        ylims = (1e-3, 30),
           lw = 4,
        label = "norm(v)", 
       xlabel = "time",
       ylabel = "norm(v)",
        title = "growth of perturbation norm",
       legend = :bottomright)

plot!(t[I], 2 * best_fit[I], # factor 2 offsets fit from curve for better visualization
           lw = 4,
        label = "best fit")
            
# The slope of the best-fit curve on a logarithmic scale approximates the rate at which instability
# grows in the simulation. Let's see how this compares with the theoretical growth rate.

println("Numerical growth rate is approximated to be ", round(slope, digits=3), ",\n",
        "which is very close to the theoretical value of 0.139.")
