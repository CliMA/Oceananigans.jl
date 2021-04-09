using Oceananigans
using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
using NCDatasets, Plots, Printf
using LinearAlgebra: norm

function norm_ũ(model)
   compute!(ũ)
   return norm(interiorparent(ũ))
end

function norm_b̃(model)
   compute!(b̃)
   return norm(interiorparent(b̃))
end

   const Ly = 100kilometers
   const Lz = 3kilometers
    const D = Lz/2
const L_jet = Ly/10
const H_jet = Lz/10 

const f  = 7.3e-5
const N² = 1e-2      # change to 1e-4
const U_max = 14.6 

const νz = 1.27e-2

grid = RegularRectilinearGrid(size = (128, 128), 
                                 y = (-Ly/2, Ly/2), z=(-Lz, 0),
                          topology = (Flat, Bounded, Bounded),
                              halo = (3, 3))

B_func(x, y, z, t, N²) = N² * (z + D)
B = BackgroundField(B_func, parameters=N²)

# Jet Profile:  
# Bickley x Gaussian [horizontal x vertical] 
ū(x, y, z) = U_max * sech(y/L_jet)^2 * exp( - (z + D)^2/H_jet^2 )
b̄(x, y, z) = U_max * tanh(y/L_jet)   * exp( - (z + D)^2/H_jet^2 ) * 2 * f * L_jet / H_jet^2 * (z + D)

perturbation(x, y, z) = randn() * sech(y/L_jet)^2 * exp( - (z + D)^2/H_jet^2 )
          uⁱ(x, y, z) = ū(x, y, z) + 1e-2 * perturbation(x, y, z)
          bⁱ(x, y, z) = b̄(x, y, z) + 1e-4 * perturbation(x, y, z)

u_forcing_func(x, y, z, t, ν) = 2 * νz / H_jet^2 * ū(x, y, z) * ( 1 - 2/H_jet^2 * (z + D)^2)
u_forcing = Forcing(u_forcing_func, parameters = νz)
          
model = IncompressibleModel(
       architecture = CPU(),
               grid = grid,
          advection = WENO5(),
        timestepper = :RungeKutta3,
           coriolis = FPlane(f=f),
            tracers = :b,
 background_fields = (b=B,),
           buoyancy = BuoyancyTracer(),
            closure = AnisotropicDiffusivity(νh=0, νz=νz),
            forcing = (u = u_forcing, ) )

set!(model, u = uⁱ, b = bⁱ)

u = model.velocities.u 
b = model.tracers.b

ũ = ComputedField(u - ū)
b̃ = ComputedField(b - b̄)

y, z = ynodes(model.velocities.u), znodes(model.velocities.u)

kwargs = (
            xlabel= "y (km)", 
            ylabel= "z (km)", 
         linewidth= 0, 
          colorbar= true,
             xlims= (-Ly/2e3, Ly/2e3), 
             ylims= (-Lz/1e3, 0)
         )
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

#wizard = TimeStepWizard(cfl=1.0, Δt=1minutes, max_change=1.1, max_Δt=2minutes)
simulation = Simulation(model, Δt=10, stop_time=2days,
                        iteration_interval=10, progress=progress)

simulation.output_writers[:fields] = JLD2OutputWriter(model, (u = ũ, b = b̃),
                                                       schedule = IterationInterval(60), 
                                                         prefix = "inertial_instability",
                                                          force = true)

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

start_time = time_ns()
run!(simulation)
finish_time = time_ns()
print("Simulation time = ", prettytime(finish_time - start_time), "\n")

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

ds2 = NCDataset(simulation.output_writers[:norms].filepath, "r")

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_u = ds2["norm_u"][:]
norm_b = ds2["norm_b"][:]

close(ds2)

norm_u_field = norm_u ./ norm_u[1]
norm_b_field = norm_b ./ norm_b[1]

plt = plot(t/hours, norm_u_field, label="u", title="Norms", xlabel="time (hours)")
plot!(plt, t/hours, norm_b_field, label="b")
savefig(plt, "tmp.png")

using Polynomials: fit

I = 9000:10000

degree = 1
linear_fit_polynomial_u = fit(t[I], log.(norm_u_field[I]), degree, var = :t)
linear_fit_polynomial_b = fit(t[I], log.(norm_b_field[I]), degree, var = :t)

constant_u, slope_u = linear_fit_polynomial_u[0], linear_fit_polynomial_u[1]
constant_b, slope_b = linear_fit_polynomial_b[0], linear_fit_polynomial_b[1]

best_fit_u = @. exp(constant_u + slope_u * t)
best_fit_b = @. exp(constant_b + slope_b * t)

plt = plot(t/hours, norm_u_field,
        yaxis = :log,
        #ylims = (2, 500),
           lw = 4,
        label = "u",
       xlabel = "time (hours)",
       ylabel = "norm(u)",
        title = "norms of perturabation",
       legend = :bottomright)

plot!(plt, t/hours, norm_b_field, lw = 4, label = "b")

plot!(plt, t[I]/hours, 2 * best_fit_u[I], lw = 4, label = "best fit u")
plot!(plt, t[I]/hours, 2 * best_fit_b[I], lw = 4, label = "best fit v")

savefig(plt, "growth.png")

# To-Do
# 2. Try wizard
# 4. Do 2D and 3D simulations on cedar (cpu and then cpu)
# 5. Try MPI?
