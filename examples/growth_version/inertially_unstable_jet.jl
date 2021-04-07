using Oceananigans
using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
using NCDatasets, Plots, Printf

   const Ly = 100kilometers
   const Lz = 3kilometers
    const D = Lz/2
const L_jet = Ly/10
const H_jet = Lz/10

grid = RegularRectilinearGrid(size=(128, 128), 
                                 y=(-Ly/2, Ly/2), z=(-Lz, 0),
                          topology=(Flat, Bounded, Bounded),
                             halo = (3, 3))

   const  f = 7.3e-5 
   const N² = 1e-2
const U_max = 14.6 

B_func(x, y, z, t, N) = N² * (z + D)
                    N = sqrt(N²)
                    B = BackgroundField(B_func, parameters=N)

model = IncompressibleModel(
     architecture = CPU(),
             grid = grid,
        advection = WENO5(),
      timestepper = :RungeKutta3,
         coriolis = FPlane(f=f),
          tracers = :b,
background_fields = (b=B,),
         buoyancy = BuoyancyTracer(),
          closure = AnisotropicDiffusivity(νh=0, νz=1.27e-2))

ū(x, y, z) = U_max * sech(y/L_jet)^2 * exp( - (z + D)^2/H_jet^2 )
b̄(x, y, z) = U_max * tanh(y/L_jet)   * exp( - (z + D)^2/H_jet^2 ) * 2 * f * L_jet / H_jet^2 * (z + D)

const ϵ = 1e-4
perturbation(x, y, z) = randn() * sech(y/L_jet)^2 * exp( - (z + D)^2/H_jet^2 )
          uⁱ(x, y, z) = ū(x, y, z) + ϵ * perturbation(x, y, z)
          bⁱ(x, y, z) = b̄(x, y, z) + ϵ * perturbation(x, y, z)

set!(model, u = uⁱ, b = bⁱ)

u = model.velocities.u 
b = model.tracers.b

ũ = ComputedField(u - ū)
b̃ = ComputedField(b - b̄)

y, z = ynodes(model.velocities.u), znodes(model.velocities.u)

kwargs = (
            xlabel="y (km)", 
            ylabel="z (km)",  
         linewidth=0, 
          colorbar=true,
             xlims=(-Ly/2e3, Ly/2e3), 
             ylims=(-Lz/1e3,0)
         )
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

#wizard = TimeStepWizard(cfl=1.0, Δt=1minutes, max_change=1.1, max_Δt=2minutes)
simulation = Simulation(model, Δt=10, stop_time=2days,
                        iteration_interval=10, progress=progress)

using LinearAlgebra: norm

function norm_ũ(model)
   compute!(ũ)
   return norm(interiorparent(ũ))
end

function norm_b̃(model)
   compute!(b̃)
   return norm(interiorparent(b̃))
end

simulation.output_writers[:fields] =
   NetCDFOutputWriter(
          model,
            (u = ũ, b = b̃),
      filepath = joinpath(@__DIR__, "inertially_unstable_jet_fields.nc"),
      schedule = IterationInterval(60),
          mode = "c")

simulation.output_writers[:norms] =
   NetCDFOutputWriter(
            model,
         (norm_u = norm_ũ, ),
        filepath = joinpath(@__DIR__, "inertially_unstable_jet_norms.nc"),
        schedule = IterationInterval(1),
      dimensions = (norm_u=(), ),
            mode = "c")

#simulation.output_writers[:norms] = JLD2OutputWriter(model, (norm_ũ = norm_ũ, norm_b̃ = norm_b̃, ),
#               filepath = joinpath(@__DIR__, "perturbation_norm_shallow_water.nc"),
#               schedule = IterationInterval(1),
#             dimensions = (perturbation_norm=(),),
#                  force = true)                                                     
                                                      
run!(simulation)

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

@info "Making a movie of perturbation zonal velocity and buoyancy..."

anim = @animate for (iter, t) in enumerate(ds["time"])

   @info "Plotting frame $iter from time $t..."

   u_snapshot = ds["u"][1, :, :, iter]
   b_snapshot = ds["b"][1, :, :, iter]

   
   u_max = maximum(abs, u_snapshot)
   u_plot = contourf(y/1e3, z/1e3, u_snapshot',
                     title = @sprintf("ũ at t = %.1f hours", t/hours),
                     color=:balance,
                     clim=(-u_max, u_max); 
                     kwargs...)

   b_max = maximum(abs, b_snapshot) 
   b_plot = contourf(y/1e3, z/1e3, b_snapshot',
                     title = @sprintf("b̃ at t = %.1f hours", t/hours),
                     color=:balance;
                     clim=(-b_max, b_max), 
                     kwargs...)

    plt = plot(u_plot, b_plot, layout=(1, 2), size=(1200, 500))

end

close(ds)

mp4(anim, "Inertial_Instability_2D.mp4", fps=15)

### Compute and plot growth rates

#using Oceananigans
#using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
#using NCDatasets, Plots, Printf

ds2 = NCDataset(simulation.output_writers[:norms].filepath, "r")
#ds2 = NCDataset("inertially_unstable_jet_norms.nc", "r")

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_u = ds2["norm_u"][:]

close(ds2)

using Polynomials: fit

I = 9000:10000

degree = 1
linear_fit_polynomial = fit(t[I], log.(norm_u[I]), degree, var = :t)

constant, slope = linear_fit_polynomial[0], linear_fit_polynomial[1]

best_fit = @. exp(constant + slope * t)

plt = plot(t/hours, norm_u,
        yaxis = :log,
        #ylims = (2, 500),
           lw = 4,
        label = "norm(u)",
       xlabel = "time (hours)",
       ylabel = "norm(u)",
        title = "norm of perturabation",
       legend = :bottomright)

plot!(plt, t[I]/hours, 2 * best_fit[I], # factor 2 offsets fit from curve for better visualization
           lw = 4,
        label = "best fit")

savefig(plt, "growth.png")

# To-Do
# 2. Try wizard
# 4. Do 2D and 3D simulations on cedar (cpu and then cpu)
# 5. Try MPI?