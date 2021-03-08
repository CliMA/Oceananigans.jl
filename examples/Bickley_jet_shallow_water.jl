using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularCartesianGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!, TimeStepWizard
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: NetCDFOutputWriter, IterationInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations
using Oceananigans.TurbulenceClosures: AnisotropicBiharmonicDiffusivity
#using Oceananigans:TimeStepper: RungeKutta3

import Oceananigans.Utils: cell_advection_timescale, prettytime

using NCDatasets, Plots, Printf, Oceananigans.Grids
using LinearAlgebra
using Polynomials
using IJulia

Lx = 2π
Ly = 20
Lz = 1
Nx = 64
Ny = 64

f  = 1
g = 9.80665

grid = RegularCartesianGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (0, Ly), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

model = ShallowWaterModel(
    architecture=CPU(),
    timestepper=:RungeKutta3,
    advection=WENO5(),
    grid=grid,
    gravitational_acceleration=g,
    coriolis=FPlane(f=f),
#    closure=AnisotropicBiharmonicDiffusivity(νh=10) 
    )

 U = 1.0
Δη = f * U / g
 ϵ = 1e-4
   
 Ω(x, y, z) = 2 * U * sech(y - Ly/2)^2 * tanh(y - Ly/2)
uⁱ(x, y, z) =  U * sech(y - Ly/2)^2 .+ ϵ * exp(- (y - Ly/2)^2 ) * randn()
ηⁱ(x, y, z) = -Δη * tanh(y - Ly/2)
hⁱ(x, y, z) = model.grid.Lz .+ ηⁱ(x, y, z)   
uhⁱ(x, y, z) = uⁱ(x, y, z) * hⁱ(x, y, z)
vhⁱ(x, y, z) = vⁱ(x, y, z) * hⁱ(x, y, z)

set!(model, uh = uhⁱ , h = hⁱ)

u_op   = model.solution.uh / model.solution.h
v_op   = model.solution.vh / model.solution.h
ω_op   = @at (Center, Center, Center) ∂x(v_op) - ∂y(u_op)
ω_pert = @at (Center, Center, Center) ω_op - Ω

ω_field = ComputedField(ω_op)
ω_pert  = ComputedField(ω_pert)

progress(sim) = @printf("Iteration: %d, time: %s, norm ω: \n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time))

simulation = Simulation(model, Δt = 1e-4, stop_iteration = 10000)

simulation.output_writers[:fields] = 
    NetCDFOutputWriter(
        model, 
        (ω = ω_field, ωp = ω_pert),
        filepath="Bickley_jet_SW.nc",
        schedule = IterationInterval(1000),
        mode = "c")

growth_rate(model) = norm(interior(v_op))
fields_to_output = (growth_rate = growth_rate,)
dims = (growth_rate=(),)
simulation.output_writers[:growth] = 
    NetCDFOutputWriter(
        model, 
        fields_to_output, 
        filepath="growth_rate_shallowwater.nc", 
        schedule=IterationInterval(1), 
        dimensions=dims,
        mode = "c")
        
        
run!(simulation)

xc = xnodes(model.solution.h)
yc = ynodes(model.solution.h)

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
)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

anim = @animate for (iter, t) in enumerate(ds["time"])
     ω = ds["ω"][:,:,1,iter]
    ωp = ds["ωp"][:,:,1,iter]

     ω_max = maximum(abs, ω)
    ωp_max = maximum(abs, ωp)

     plot_ω = contour(xc, yc, ω',  clim=(-ω_max,  ω_max),  title=@sprintf("Total ζ at t = %.3f", t); kwargs...)
    plot_ωp = contour(xc, yc, ωp', clim=(-ωp_max, ωp_max), title=@sprintf("Perturbation ζ at t = %.3f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1,2), size=(1200, 500))

    print("At t = ", t, " maximum of ωp = ", maximum(abs, ωp), "\n")
end

close(ds)

mp4(anim, "Bickley_Jet_ShallowWater.mp4", fps=15)

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

iterations = keys(ds2["time"])

t = ds2["time"][:]
σ = ds2["growth_rate"][:]

close(ds2)

#I = 50000:60000
I = 40:100 
best_fit = fit((t[I]), log.(σ[I]), 1)
poly = 2 .* exp.(best_fit[0] .+ best_fit[1]*t[I])

plt = plot(t[1000:end], log.(σ[1000:end]), lw=4, label="sigma", title="growth rate", legend=:bottomright)
plot!(plt, t[I], log.(poly), lw=4, label="best fit")
savefig(plt, "growth_rates.png")

print("Best slope = ", best_fit[1])
