using Oceananigans
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.Fields, Oceananigans.AbstractOperations
using Oceananigans.OutputWriters: NetCDFOutputWriter, IterationInterval

using NCDatasets, Plots, Printf, Oceananigans.Grids
using LinearAlgebra
using Polynomials
using IJulia

Lx = 6.61
Ly = 20
Lz = 1
Nx = 256
Ny = 256

f  = 1

grid = RegularCartesianGrid(size = (Nx, Ny, 1),
                            x = (0, Lx), y = (0, Ly), z = (0, Lz),
                            topology = (Periodic, Bounded, Bounded))

model = HydrostaticFreeSurfaceModel(
    grid=grid, 
    coriolis=FPlane(f=f)
    )

 U = 1.0
 g = model.free_surface.gravitational_acceleration
Δη = f * U / g
 ϵ = 1e-2

 Ω(x, y, z) = 2 * U * sech(y - Ly/2)^2 * tanh(y - Ly/2)
uⁱ(x, y, z) =  U * sech(y - Ly/2)^2 .+ ϵ * exp(- (y - Ly/2)^2 ) * randn()
ηⁱ(x, y, z) = -Δη * tanh(y - Ly/2)

set!(model, u=uⁱ, η=ηⁱ)

 u_op = model.velocities.u
 v_op = model.velocities.v
 ω_op = @at (Face, Center, Center) ∂x(v_op) - ∂y(u_op)
ω_pert = @at (Face, Center, Center) ω_op - Ω

ω_field = ComputedField(ω_op)
ω_pert  = ComputedField(ω_pert)


simulation = Simulation(model, Δt = 1e-3, stop_iteration = 150000)

simulation.output_writers[:fields] = 
    NetCDFOutputWriter(
        model, 
        (ω = ω_field, ωp = ω_pert),
        filepath="Bickley_jet_HY.nc",
        schedule = IterationInterval(1000),
        mode = "c")

growth_rate(model) = norm(interior(model.velocities.v))
fields_to_output = (growth_rate = growth_rate,)
dims = (growth_rate=(),)
simulation.output_writers[:growth] = 
    NetCDFOutputWriter(
        model, 
        fields_to_output, 
        filepath="growth_rate_hydrostatic.nc", 
        schedule=IterationInterval(1), 
        dimensions=dims,
        mode = "c")

run!(simulation)

xc = xnodes(model.free_surface.η)
yc = ynodes(model.free_surface.η)

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

mp4(anim, "Bickley_Jet_HydrostaticFreeSurface.mp4", fps = 15)

ds2 = NCDataset(simulation.output_writers[:growth].filepath, "r")

iterations = keys(ds2["time"])

t = ds2["time"][:]
σ = ds2["growth_rate"][:]

close(ds2)

I = 50000:60000
best_fit = fit((t[I]), log.(σ[I]), 1)
poly = 2 .* exp.(best_fit[0] .+ best_fit[1]*t[I])

plt = plot(t[1000:end], log.(σ[1000:end]), lw=4, label="sigma", title="growth rate", legend=:bottomright)
plot!(plt, t[I], log.(poly), lw=4, label="best fit")
savefig(plt, "growth_rates.png")

print("Best slope = ", best_fit[1])
