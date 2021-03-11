# # Shallow Water Example: an unstable Bickley jet 
#
# This example shows how to use `Oceananigans.ShallowWwaterModel` to simulate
# the evolution of an unstable, geostrophically balanced, Bickley jet.
# The model solves the governing equations for the shallow water model in
# conservation form.  The geometry is that of a periodic channel
# in the ``x``-direction with a flat bottom and a free-surface.  The initial
# conditions are that of a Bickley jet with small amplitude perturbations.
# The interested reader can see ["The nonlinear evolution of barotropically unstable jets," J. Phys. Oceanogr. (2003)](https://doi.org/10.1175/1520-0485(2003)033<2173:TNEOBU>2.0.CO;2)
# for more details on this specific problem. 
#
# Unlike the other models, the fields that are simulated are the mass transports, 
# ``uh`` and ``vh`` in the ``x`` and ``y`` directions, respectively,
# and the height ``h``.  Note that ``u`` and ``v`` are the velocities, which 
# can easily be computed when needed.
#
# ## Install dependencies
#
# First we make sure that we have all of the packages that are required to
# run the simulation.
# ```julia
# using Pkg
# pkg"add Oceananigans, NCDatasets, Plots, Printf, LinearAlgebra, Polynomials"
# ```
#
# FJP: 1] Change the title in sidebar to be something else.
#
# FJP: 2] Need to reduce these lines here as this is way too much!

using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids
using Oceananigans.Grids: Periodic, Bounded, RegularRectilinearGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: NetCDFOutputWriter, TimeInterval, IterationInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations

import Oceananigans.Utils: cell_advection_timescale, prettytime

# ## Two-dimensional domain 
#
# The shallow water model is a two-dimensional model and we must specify
# the number of vertical grid points to be one.  
# We pick the length of the domain to fit one unstable mode along the channel
# exactly.  Note that ``Lz`` is the mean depth of the fluid.  It is determined
# using linear stability theory that the wavenumber that yields the most unstable
# mode is ``k=0.95``. See [Linear-Stability-Calculators](https://github.com/francispoulin/Linear-Stability-Calculators/tree/main/ShallowWater/Julia/Cartesian)
# for details on how that is computed.

Lx = 2π/0.95
Ly = 20
Lz = 1
Nx = 128
Ny = Nx

grid = RegularRectilinearGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (0, Ly), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

# ## Physical parameters
#
# This is a toy problem and we choose the parameters so this jet idealizes
# a relatively narrow mesosale jet.   
# The physical parameters are
#
# FJP: 3] Should we change the parameters to be on planetary scales?
# 
#   * ``f``: Coriolis parameter
#   * ``g``: Acceleration due to gravity
#   * ``U``: Maximum jet speed
#   * ``\Delta\eta``: Maximum free-surface deformation that is dictated by geostrophy
#   * ``\epsilon`` : Amplitude of the perturbation

f = 1;
g = 9.80665;
U = 1.0;
Δη = f * U / g;
ϵ = 1e-4;

# ## Building a `ShallowWaterModel`
#
# We use `grid`, `coriolis` and `gravitational_acceleration` to build the model.
# Furthermore, we specify this runs on `CPUs`, uses `RungeKutta3` for time-stepping
# and `WENO5` for advection.

model = ShallowWaterModel(
    architecture=CPU(),
    timestepper=:RungeKutta3,
    advection=WENO5(),
    grid=grid,
    gravitational_acceleration=g,
    coriolis=FPlane(f=f),
    )

# ## Background state and perturbation
# 
# We specify `Ω` to be the background vorticity of the jet in the absence of any perturbations.
# The initial conditions have a small perturbation that is random in space and 
# decays away from the center of the jet.

  Ω(x, y, z) = 2 * U * sech(y - Ly/2)^2 * tanh(y - Ly/2);
 uⁱ(x, y, z) =   U * sech(y - Ly/2)^2 .+ ϵ * exp(- (y - Ly/2)^2 ) * randn();
 ηⁱ(x, y, z) = -Δη * tanh(y - Ly/2);
 hⁱ(x, y, z) = model.grid.Lz .+ ηⁱ(x, y, z);
uhⁱ(x, y, z) = uⁱ(x, y, z) * hⁱ(x, y, z);

# We set the initial conditions for the zonal mass transport `uhⁱ` and height `hⁱ`.

set!(model, uh = uhⁱ , h = hⁱ)

# We compute the total vorticity and the perturbation vorticity.

uh, vh, h = model.solution
        u = ComputedField(uh / h)
        v = ComputedField(vh / h)
  ω_field = ComputedField( ∂x(vh/h) - ∂y(uh/h) )
   ω_pert = ComputedField( ω_field - Ω )

# Progress will output the clock times and the norm of the cross channel velocity,
# as this is used to compute the growth rates.  We obtain the `norm` function from `LinearAlgebra`.

using LinearAlgebra, Printf

function progress(sim)
    compute!(v)
    @printf("Iteration: %d, time: %s, norm v: %f\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    norm(interior(v)) )
end

# ## Running a `Simulation`
#
# We pick the time-step that ensures to resolve the surface gravity waves.
# A time-step wizard can be applied to use an adaptive time step.

simulation = Simulation(model, Δt = 1e-2, stop_time = 150.0)

# ## Prepare output files
#
# Define a function to compute the growth rate based on the cross channel velocity

function growth_rate(model)
    compute!(v)
    return norm(interior(v))
end

# Choose the two fields to be output to be the total and perturbation vorticity.

outputs = (ω_total = ω_field, ω_pert = ω_pert)

# Build the `output_writer` for the two-dimensional fields to be output.
# Output every `t = 1.0`.

simulation.output_writers[:fields] =
    NetCDFOutputWriter(
        model,
        (ω = ω_field, ωp = ω_pert),
        filepath=joinpath(@__DIR__, "Bickley_jet_shallow_water.nc"),
        schedule=TimeInterval(1.0),
        mode = "c")

# Build the `output_writer` for the growth rate, which is a scalar field.
# Output every time step.
#
# FJP 4] Is there a better way to produce a diagnostics file like this?

simulation.output_writers[:growth] =
    NetCDFOutputWriter(
        model,
        (growth_rate = growth_rate,),
        filepath="growth_rate_shallow_water.nc",
        schedule=IterationInterval(1),
        dimensions=(growth_rate=(),),
        mode = "c")

run!(simulation)

# ## Visiualize the results
#

using NCDatasets, Plots, IJulia;

# Define the coordinates for plotting

xf = xnodes(ω_field);
yf = ynodes(ω_field);

# Define keyword arguments for plotting the contours

kwargs = (
         xlabel = "x",
         ylabel = "y",
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           ylim = (0, Ly),
           xlim = (0, Lx)
);

# Read in the `output_writer` for the two-dimensional fields
# and then create an animation showing both the total and perturbation
# vorticities.

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

anim = @animate for (iter, t) in enumerate(ds["time"])
        ω = ds["ω"][:,:,1,iter]
    ωp = ds["ωp"][:,:,1,iter]

     ω_max = maximum(abs, ω)
    ωp_max = maximum(abs, ωp)

     plot_ω = contour(xf, yf, ω',  clim=(-ω_max,  ω_max),  title=@sprintf("Total ω at t = %.3f", t); kwargs...)
    plot_ωp = contour(xf, yf, ωp', clim=(-ωp_max, ωp_max), title=@sprintf("Perturbation ω at t = %.3f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1,2), size=(1200, 500))

    print("At t = ", t, " maximum of ωp = ", maximum(abs, ωp), "\n")
end

close(ds)

mp4(anim, "Bickley_Jet_ShallowWater.mp4", fps=15)

# Read in the `output_writer` for the scalar field.

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

iterations = keys(ds2["time"])

t = ds2["time"][:]
σ = ds2["growth_rate"][:]

close(ds2)

# Import `Polynomials` to be able to use `best_fit`.
# Compute the best fit slope and save the figure.

using Polynomials

I = 6000:7000
best_fit = fit((t[I]), log.(σ[I]), 1)
poly = 2 .* exp.(best_fit[0] .+ best_fit[1]*t[I])

plt = plot(t[1000:end], log.(σ[1000:end]), lw=4, label="sigma", 
        xlabel="time", ylabel="log(v)", title="growth rate", legend=:bottomright)
plot!(plt, t[I], log.(poly), lw=4, label="best fit")
savefig(plt, "growth_rates.png")

print("Best slope = ", best_fit[1])