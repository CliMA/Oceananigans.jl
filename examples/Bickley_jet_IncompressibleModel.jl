using Oceananigans

Lx = 2π
Ly = 20
Lz = 1
Nx = 256
Ny = 256

f  = 1

grid = RegularCartesianGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (0, Ly), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

using Oceananigans.Models: IncompressibleModel
using Oceananigans.Fields, Oceananigans.AbstractOperations

model = IncompressibleModel(
    grid=grid, 
    coriolis=FPlane(f=f)
    )

 U = 1.0
 f = model.coriolis.f
 ϵ = 1e-2

  Ω(x, y, z) = 2 * U * sech(y - Ly/2)^2 * tanh(y - Ly/2)
 uⁱ(x, y, z) =     U * sech(y - Ly/2)^2 .+ ϵ * exp(- (y - Ly/2)^2 ) * randn()

set!(model, u=uⁱ)

u_op = model.velocities.u
v_op = model.velocities.v
ω_op = @at (Face, Center, Center) ∂x(v_op) - ∂y(u_op)
ω_pert = @at (Face, Center, Center) ω_op - Ω

u_field = ComputedField(u_op)
ω_field = ComputedField(ω_op)
ω_pert  = ComputedField(ω_pert)

simulation = Simulation(model, Δt = 1e-2, stop_iteration = 15000)

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation.output_writers[:fields] = 
    JLD2OutputWriter(
        model, 
        (u = u_field, ω = ω_field, ωp = ω_pert), 
        schedule = IterationInterval(100),
        prefix = "Bickley_Jet",
        force = true)

run!(simulation)

using JLD2, Plots, Printf, Oceananigans.Grids
using LinearAlgebra
using IJulia

xc = xnodes(model.velocities.u)
yc = ynodes(model.velocities.u)

kwargs = (
         xlabel = "x",
         ylabel = "y",
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
           xlim = (0, Ly),
           ylim = (0, Lx)
)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

     t = file["timeseries/t/$iter"]
     ω = file["timeseries/ω/$iter"][:, :, 1]
    ωp = file["timeseries/ωp/$iter"][:, :, 1]

     ω_max = maximum(abs, ω)
    ωp_max = maximum(abs, ωp)

     plot_ω = contour(yc, xc, ω,  clim=(-ω_max,  ω_max),  title=@sprintf("Total ζ at t = %.3f", t); kwargs...)
    plot_ωp = contour(yc, xc, ωp, clim=(-ωp_max, ωp_max), title=@sprintf("Perturbation ζ at t = %.3f", t); kwargs...)

    plot(plot_ω, plot_ωp, layout = (1,2), size=(1200, 500))

    print("At t = ", t, " maximum of ωp = ", maximum(abs, ωp), "\n")
end

close(file)

mp4(anim, "Bickley_Jet_IncompressibleModel.mp4", fps = 15) # hide