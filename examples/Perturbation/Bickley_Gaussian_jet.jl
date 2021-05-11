using Oceananigans
using Oceananigans.Units: hours, days, seconds, meters, kilometers, minutes
using NCDatasets, Plots, Printf
using LinearAlgebra: norm

function norm_u(model)
   compute!(u)
   return norm(interiorparent(u))
end

function norm_b(model)
   compute!(b)
   return norm(interiorparent(b))
end

const Ly = 100kilometers
const Lz = 3kilometers

parameters = (  f = 7.3e-5,          # Coriolis frequency
                N = 1e-1,            # stratification frequency      *** change to 1e-2
               νz = 1.27e-2,         # vertical viscosity
                L = Ly/10,           # jet length
                H = Lz/10,           # jet depth
                D = Lz/2,            # jet depth
                U = 14.6)            # jet speed

grid = RegularRectilinearGrid(size = (128, 128), 
                                 y = (-Ly/2, Ly/2), z=(-Lz, 0),
                          topology = (Flat, Bounded, Bounded),
                              halo = (3, 3))

# Basic State
U_func(x, y, z, t, p) = p.U * sech(y/p.L)^2 * exp( - (z + p.D)^2/p.H^2 )
B_func(x, y, z, t, p) = p.U * tanh(y/p.L)   * exp( - (z + p.D)^2/p.H^2 ) * 2 * p.f * p.L / p.H^2 * (z + p.D) + p.N^2 * (z + p.D)

B_field = BackgroundField(B_func, parameters = parameters)
U_field = BackgroundField(U_func, parameters = parameters)

# Initial Conditions
uⁱ(x, y, z) = 1e-2 * randn() * sech(y/parameters.L)^2 * exp( - (z + parameters.D)^2/parameters.H^2 )
bⁱ(x, y, z) = 1e-4 * randn() * sech(y/parameters.L)^2 * exp( - (z + parameters.D)^2/parameters.H^2 )

model = IncompressibleModel(
       architecture = CPU(),
               grid = grid,
          advection = WENO5(),
        timestepper = :RungeKutta3,
           coriolis = FPlane(f = parameters.f),
            tracers = :b,
  background_fields = (u=U_field, b=B_field),
           buoyancy = BuoyancyTracer(),
            closure = AnisotropicDiffusivity(νh=0, νz=parameters.νz) )

set!(model, u = uⁱ, b = bⁱ)

u = ComputedField(model.velocities.u)
b = ComputedField(model.tracers.b)

xu, yu, zu = nodes(model.velocities.u)
xb, yb, zb = nodes(model.tracers.b)

kwargs = (
            xlabel= "y (km)", 
            ylabel= "z (km)", 
         linewidth= 0, 
          colorbar= true,
             xlims= (-Ly/2/kilometers, Ly/2/kilometers), 
             ylims= (-Lz/kilometers, 0)
         )
progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt))

#wizard = TimeStepWizard(cfl=1.0, Δt=1minutes, max_change=1.1, max_Δt=2minutes)
simulation = Simulation(model, Δt=10, stop_time=0.5hours,
                        iteration_interval=60, progress=progress)

simulation.output_writers[:fields] =
   NetCDFOutputWriter(
          model,
            (u = u, b = b),
      filepath = joinpath(@__DIR__, "inertially_unstable_jet_fields.nc"),
      schedule = IterationInterval(60),
          mode = "c")

simulation.output_writers[:norms] =
   NetCDFOutputWriter(
            model,
         (norm_u = norm_u, norm_b = norm_b),
        filepath = joinpath(@__DIR__, "inertially_unstable_jet_norms.nc"),
        schedule = IterationInterval(1),
      dimensions = (norm_u=(), norm_b=()),
            mode = "c")

start_time = time_ns()
run!(simulation)
finish_time = time_ns()
print("Simulation time = ", prettytime(finish_time - start_time), "\n")

# Make animation 

ds = NCDataset(simulation.output_writers[:fields].filepath, "r")

iterations = keys(ds["time"])

@info "Making a movie of perturbation zonal velocity and buoyancy..."

anim = @animate for (iter, t) in enumerate(ds["time"])

   @info "Plotting frame $iter from time $t..."

   u_snapshot = ds["u"][1, :, :, iter]
   b_snapshot = ds["b"][1, :, :, iter]

   u_max = maximum(abs, u_snapshot)
   u_plot = contourf(yu/kilometers, zu/kilometers, u_snapshot',
                     title = @sprintf("u at t = %.1f hours", t/hours),
                     color=:balance,
                     clim=(-u_max, u_max);
                     kwargs...)

   b_max = maximum(abs, b_snapshot)
   b_plot = contourf(yb/kilometers, zb/kilometers, b_snapshot',
                     title = @sprintf("b at t = %.1f hours", t/hours),
                     color=:balance;
                     clim=(-b_max, b_max),
                     kwargs...)

    plt = plot(u_plot, b_plot, layout=(1, 2), size=(1200, 500))

end

close(ds)

mp4(anim, "Inertial_Instability_2D.mp4", fps=15)

# Compute growth rates and plot

ds2 = NCDataset(simulation.output_writers[:norms].filepath, "r")

iterations = keys(ds2["time"])

     t = ds2["time"][:]
norm_u_field = ds2["norm_u"][:]
norm_b_field = ds2["norm_b"][:]

close(ds2)

norm_u_field = norm_u_field ./ norm_u_field[1]
norm_b_field = norm_b_field ./ norm_b_field[1]

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
