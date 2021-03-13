# # An unstable Bickley jet in Shallow Water model 
#
# This example shows how to use `Oceananigans.ShallowWaterModel` to simulate
# the evolution of an unstable, geostrophically balanced, Bickley jet.
# The model solves the governing equations for the shallow water model in
# conservative form.  The geometry is that of a periodic channel
# in the ``x``-direction with a flat bottom and a free-surface.  The initial
# conditions are that of a Bickley jet with small amplitude perturbations.
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
#

using Oceananigans
using Oceananigans.Models: ShallowWaterModel

# ## Two-dimensional domain 
#
# The shallow water model is a two-dimensional model and thus the number of vertical
# points `Nz` must be set to one.  Note that ``L_z`` is the mean depth of the fluid. 

Lx = 2π
Ly = 20
Lz = 1
Nx = 128
Ny = Nx

grid = RegularRectilinearGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (-Ly/2, Ly/2), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

# ## Physical parameters
#
# This is a toy problem and we choose the parameters so this jet idealizes
# a relatively narrow mesoscale jet.   
# The physical parameters are
#
#   * ``f``: Coriolis parameter
#   * ``g``: Acceleration due to gravity
#   * ``U``: Maximum jet speed
#   * ``\Delta\eta``: Maximum free-surface deformation that is dictated by geostrophy
#   * ``\epsilon`` : Amplitude of the perturbation

f = 1
g = 9.80665
U = 1.0
Δη = f * U / g
ϵ = 1e-4
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
# The background velocity ``\overline{u}`` and free-surface ``\overline{\eta}`` are chosen to 
# represent a geostrophically balanced Bickely jet with maximum speed of ``U`` and 
# maximum free-surface deformation of ``Δη``, i.e.,
#
# ```math
# \begin{align}
# \overline{\eta}(y) & = - Δη \tanh(y) ,
# \overline{u}(y) & = U \mathrm{sech}^2(y) .
# \end{align}
# ```
# 
# The vorticity of the background state is ``Ω = - \partial_y \overline{u} = 2 U \mathrm{sech}^2(y) \tanh(y)``.
#
# Linear stability theory predicts that for the particular parameters that we consider here,
# the growth rate for the maximum growth rate should be ``0.14``.
# 
# We specify `Ω` to be the background vorticity of the jet in the absence of any perturbations.
# The initial conditions include a small-ampitude perturbation that decays away from the center
# of the jet.

  Ω(x, y, z) = 2 * U * sech(y)^2 * tanh(y)
 uⁱ(x, y, z) =   U * sech(y)^2 + ϵ * exp(- y^2 ) * randn()
 ηⁱ(x, y, z) = -Δη * tanh(y)
 hⁱ(x, y, z) = model.grid.Lz + ηⁱ(x, y, z)
uhⁱ(x, y, z) = uⁱ(x, y, z) * hⁱ(x, y, z)
nothing # hide

# We set the initial conditions for the zonal mass transport `uhⁱ` and height `hⁱ`.

set!(model, uh = uhⁱ , h = hⁱ)

# We compute the total vorticity and the perturbation vorticity.

uh, vh, h = model.solution
        u = ComputedField(uh / h)
        v = ComputedField(vh / h)
        v = ComputedField(v)
        ω = ComputedField(∂x(vh/h) - ∂y(uh/h))
   ω_pert = ComputedField(ω - Ω)
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

using LinearAlgebra, Printf

function perturbation_norm(model)
    compute!(v)
    return norm(interior(v))
end
nothing # hide

# Choose the two fields to be output to be the total and perturbation vorticity.

outputs = (ω_total = ω, ω_pert = ω_pert)

# Build the `output_writer` for the two-dimensional fields to be output.
# Output every `t = 1.0`.

simulation.output_writers[:fields] =
    NetCDFOutputWriter(
        model,
        (ω = ω, ωp = ω_pert),
        filepath=joinpath(@__DIR__, "Bickley_jet_shallow_water.nc"),
        schedule=TimeInterval(1.0),
        mode = "c")

# Build the `output_writer` for the growth rate, which is a scalar field.
# Output every time step.
#

simulation.output_writers[:growth] =
    NetCDFOutputWriter(
        model,
        (perturbation_norm = perturbation_norm,),
        filepath=joinpath(@__DIR__, "perturbation_norm_shallow_water.nc"),
        schedule=IterationInterval(1),
        dimensions=(perturbation_norm=(),),
        mode = "c")

run!(simulation)

# ## Visualize the results
#

using NCDatasets
nothing # hide

# Define the coordinates for plotting

xf = xnodes(ω)
yf = ynodes(ω)
nothing # hide

# Define keyword arguments for plotting the contours

kwargs = (
         xlabel = "x",
         ylabel = "y",
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           ylim = (-Ly/2, Ly/2),
           xlim = (0, Lx)
)
nothing # hide

# Read in the `output_writer` for the two-dimensional fields
# and then create an animation showing both the total and perturbation
# vorticities.

using Plots

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

anim = @animate for (iter, t) in enumerate(ds["time"])
     ω = ds["ω"][:, :, 1, iter]
    ωp = ds["ωp"][:, :, 1, iter]

     ω_max = maximum(abs, ω)
    ωp_max = maximum(abs, ωp)

     plot_ω = contour(xf, yf, ω',  clim=(-1,  1),  title=@sprintf("Total ω at t = %.3f", t); kwargs...)
    plot_ωp = contour(xf, yf, ωp', clim=(-ωp_max, ωp_max), title=@sprintf("Perturbation ω at t = %.3f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1,2), size=(1200, 500))
end

close(ds)

mp4(anim, "Bickley_Jet_ShallowWater.mp4", fps=15)

# Read in the `output_writer` for the scalar field.

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_v = ds2["perturbation_norm"][:]

close(ds2)

# We import the `fit` function from `Polynomials.jl` to compute the best-fit slope of the 
# perturbation norm on a logarithmic plot. This slope corresponds to the growth rate.

using Polynomials: fit

I = 6000:7000

degree = 1
linear_fit_polynomial = fit(t[I], log.(norm_v[I]), degree)

best_fit = exp.(linear_fit_polynomial[0] .+ linear_fit_polynomial[1] * t)

plt = plot(t, norm_v,
            yaxis = :log,
            ylims = (1e-3, 30),
               lw = 4,
            label = "norm(v)", 
           xlabel = "time",
           ylabel = "norm(v)",
            title = "growth of perturbation norm",
           legend = :bottomright)
plot!(plt, t[I], 2 * best_fit[I], # factor 2 offsets our fit from the curve for better visualization
               lw = 4,
            label = "best fit")
            
# We can compute the slope of the curve on a log scale, which approximates the growth rate
# of the simulation. This should be close to the theoretical prediction.

println("Growth rate in the simulation is approximated to be ", linear_fit_polynomial[1], ",\n",
        "which is close to the theoretical value of 0.14.")