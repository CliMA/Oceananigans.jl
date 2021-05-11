using Oceananigans
using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
using NCDatasets, Plots, Printf
using Revise

global_parameters = ( Ly = 100kilometers,          # domain length
                      Lz = 3kilometers,            # domain depth
                      f  = 7.3e-5,                 # Corioilis parameter
                      N  = 1e-2)                   # Buoyancy frequency squared

jet_parameters = ( Ly = global_parameters.Ly/10,   # jet length
                   Lz = global_parameters.Lz/10,   # jet depth
                   D  = global_parameters.Lz/2,    # jet depth
                   U  = 14.6)                      # jet speed

grid = RegularRectilinearGrid(size = (8, 8),
                                 y = (-global_parameters.Ly/2, global_parameters.Ly/2),
                                 z = (-global_parameters.Lz, 0),
                          topology = (Flat, Bounded, Bounded),
                              halo = (3, 3))

# FJP: include jet here too?
# FJP: define B_jet and U_jet maybe and put in BackgroundFields?
B_func(x, y, z, t, N) = N^2 * (z + jet_parameters.D)
          Bbackground = BackgroundField(B_func, parameters=global_parameters.N)

# FJP: write a function that takes the inputs of structure in the horizontal and vertical,
# as well as parameters. Should spit out ū and b̄.

function Bickley_Gaussian_jet(jet_parameters, global_parameters)

      Ly = jet_parameters.Ly
      Lz = jet_parameters.Lz
       D = jet_parameters.D 
   U_max =     jet_parameters.U
   B_max = 2 * jet_parameters.U * global_parameters.f * jet_parameters.Ly / jet_parameters.Lz^2

   # one-dimensional structures of the jet
   Horizontal_u(x, y, z) = sech(y/Ly)^2
   Horizontal_b(x, y, z) = tanh(y/Ly)
     Vertical_u(x, y, z) = exp( - (z + D)^2/Lz^2 )
     Vertical_b(x, y, z) = tanh(y/Ly) * exp( - (z + D)^2/Lz^2 )

   # basic state profiles
   ū(x, y, z) = U_max * Horizontal_u(x, y, z) * Vertical_u(x, y, z)
   b̄(x, y, z) = B_max * Horizontal_b(x, y, z) * Vertical_b(x, y, z)

   return ū, b̄
end

ū2(x, y, z), b̄2(x, y, z) = Bickley_Gaussian_jet(jet_parameters, global_parameters)

# one-dimensional structures of the jet
Horizontal_u(x, y, z) = sech(y/jet_parameters.Ly)^2
Horizontal_b(x, y, z) = tanh(y/jet_parameters.Ly)
  Vertical_u(x, y, z) = exp( - (z + jet_parameters.D)^2/jet_parameters.Lz^2 )
  Vertical_b(x, y, z) = tanh(y/jet_parameters.Ly) * exp( - (z + jet_parameters.D)^2/jet_parameters.Lz^2 )

# amplitudes of the jet
U_maximum  = jet_parameters.U
B_maximum  = jet_parameters.U * 2 * global_parameters.f * jet_parameters.Ly / jet_parameters.Lz^2

# basic state profiles
ū(x, y, z) = U_maximum * Horizontal_u(x, y, z) * Vertical_u(x, y, z)
b̄(x, y, z) = B_maximum * Horizontal_b(x, y, z) * Vertical_b(x, y, z)

              const ϵᵤ = 1e-2
              const ϵᵦ = 1e-4
perturbation(x, y, z) = randn() * Hᵤ(x, y, z) * Zᵤ(x, y, z)
          uⁱ(x, y, z) = ū(x, y, z) + ϵᵤ * perturbation(x, y, z)
          bⁱ(x, y, z) = b̄(x, y, z) + ϵᵦ * perturbation(x, y, z)

const ν = 0.0
u_forcing_func(x, y, z, t, ν) = ν * ū(x, y, z)
u_forcing = Forcing(u_forcing_func, parameters=ν)

model = IncompressibleModel(
            architecture = GPU(),
                    grid = grid,
               advection = WENO5(),
             timestepper = :RungeKutta3,
                coriolis = FPlane(f=global_parameters.f),
                 tracers = :b,
       background_fields = (b=Bbackground,),
                buoyancy = BuoyancyTracer(),
                 closure = AnisotropicDiffusivity(νh=0, νz=1.27e-2),
                 forcing = (u = u_forcing, ) )

set!(model, u = uⁱ, b = bⁱ)

u = model.velocities.u
b = model.tracers.b

ũ = ComputedField(u - ū)
b̃ = ComputedField(b - b̄)

xu, yu, zu = nodes(model.velocities.u)
xb, yb, zb = nodes(model.tracers.b)

kwargs = (
            xlabel="y (km)",
            ylabel="z (km)",
         linewidth=0,
          colorbar=true,
             xlims=(-global_parameters.Ly/kilometers, global_parameters.Ly/kilometers),
             ylims=(-global_parameters.Lz/kilometers, 0)
         )
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

#wizard = TimeStepWizard(cfl=1.0, Δt=1minutes, max_change=1.1, max_Δt=2minutes)
simulation = Simulation(model, Δt=10, stop_time=2800,
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
         (norm_u = norm_ũ, norm_b = norm_b̃),
        filepath = joinpath(@__DIR__, "inertially_unstable_jet_norms.nc"),
        schedule = IterationInterval(1),
      dimensions = (norm_u=(), norm_b=()),
            mode = "c")

#=
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

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_u = ds2["norm_u"][:]
norm_b = ds2["norm_b"][:]

close(ds2)

#plt = plot(t, norm_b)
#plot!(plt, t, norm_b)
#savefig(plt, "tmp.png")

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
=#
# To-Do
# 2. Try wizard
# 4. Do 2D and 3D simulations on cedar (cpu and then cpu)
# 5. Try MPI?