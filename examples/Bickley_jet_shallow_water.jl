using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularRectilinearGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!, TimeStepWizard
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: NetCDFOutputWriter, TimeInterval, IterationInterval
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
Nx = 256
Ny = Nx

 f = 1
 g = 9.80665
 U = 1.0
Δη = f * U / g
 ϵ = 1e-4

grid = RegularRectilinearGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (0, Ly), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

model = ShallowWaterModel(
    architecture=CPU(),
    timestepper=:RungeKutta3,
    advection=WENO5(),
    grid=grid,
    gravitational_acceleration=g,
    coriolis=FPlane(f=f),
    )

   
 Ω(x, y, z) = 2 * U * sech(y - Ly/2)^2 * tanh(y - Ly/2)
uⁱ(x, y, z) =   U * sech(y - Ly/2)^2 .+ ϵ * exp(- (y - Ly/2)^2 ) * randn()
ηⁱ(x, y, z) = -Δη * tanh(y - Ly/2)
hⁱ(x, y, z) = model.grid.Lz .+ ηⁱ(x, y, z)   
uhⁱ(x, y, z) = uⁱ(x, y, z) * hⁱ(x, y, z)

set!(model, uh = uhⁱ , h = hⁱ)

uh, vh, h = model.solution
        u = ComputedField(uh / h)
        v = ComputedField(vh / h)
  ω_field = ComputedField( ∂x(vh/h) - ∂y(uh/h) )
   ω_pert = ComputedField( ω_field - Ω )

function progress(sim)  
    compute!(v)
    @printf("Iteration: %d, time: %s, norm v: %f\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    norm(interior(v)) )
end

simulation = Simulation(model, Δt = 1e-3, stop_time = 150.0, progress=progress)

function growth_rate(model)
    compute!(v)
    return norm(interior(v))
end

outputs = (ω_total = ω_field, ω_pert = ω_pert)

simulation.output_writers[:fields] = 
    NetCDFOutputWriter(
        model, 
        (ω = ω_field, ωp = ω_pert),
        filepath="Bickley_jet_shallow_water.nc",
        schedule=TimeInterval(1.0),         
        mode = "c")

growth_rate(model) = norm(interior(v))

simulation.output_writers[:growth] = 
    NetCDFOutputWriter(
        model, 
        (growth_rate = growth_rate,),
        filepath="growth_rate_shallow_water.nc", 
        schedule=IterationInterval(1), 
        dimensions=(growth_rate=(),),
        mode = "c")

run!(simulation)

xf = xnodes(ω_field)
yf = ynodes(ω_field)

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

     plot_ω = contour(xf, yf, ω',  clim=(-ω_max,  ω_max),  title=@sprintf("Total ζ at t = %.3f", t); kwargs...)
    plot_ωp = contour(xf, yf, ωp', clim=(-ωp_max, ωp_max), title=@sprintf("Perturbation ζ at t = %.3f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1,2), size=(1200, 500))

    print("At t = ", t, " maximum of ωp = ", maximum(abs, ωp), "\n")
end

close(ds)

mp4(anim, "Bickley_Jet_ShallowWater.mp4", fps=15)
#gif(anim, "Bickley_Jet_ShallowWater.gif", fps=15)

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
