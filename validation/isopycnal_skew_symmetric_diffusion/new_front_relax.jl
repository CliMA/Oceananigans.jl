using Printf
using Statistics
using Random
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode
using GLMakie
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: TriadIsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: FluxTapering

gradient = "y"
filename = "new_front_relax_" * gradient

# Domain
dz=[50, 50, 55, 60, 65, 70, 80, 95, 120, 155, 200, 260, 320, 400, 480]
Ly = 320kilometers  # north-south extent [m]
Lz = sum(dz)        # depth [m]
Nx = 3
Ny = 32
Nz = 15
#save_fields_interval = 1day
stop_time = 30days
save_fields_interval = 1days
Δt = 15minutes

# viscosity
viscAh=300.
viscAz=2.e-4

grid = RectilinearGrid(topology = (Flat, Bounded, Bounded),
                       size = (Ny, Nz), 
                       y = (-Ly/2, Ly/2),
                       z = ([0 cumsum(reverse(dz))']'.-Lz),
                       halo = (3, 3))

#coriolis = FPlane(latitude = -45)
coriolis = FPlane( f = 1.e-4)

h_visc= HorizontalScalarDiffusivity( ν=viscAh )
v_visc= VerticalScalarDiffusivity( ν=viscAz )

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)

triad_closure = TriadIsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), κ_skew = 0e3,
                                                       κ_symmetric = 1e3,
                                                       slope_limiter = gerdes_koberle_willebrand_tapering)

cox_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew = 0e3,
                                                κ_symmetric = 1e3,
                                                slope_limiter = gerdes_koberle_willebrand_tapering)

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                    closure = triad_closure, #, h_visc, v_visc),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c))

@info "Built $model."

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification, corresponds to N/f = 20
M² = 4e-9 # [s⁻²] horizontal buoyancy gradienta, gives a slope of 1.e-3

# tracer patch centered on yC,zC:
yC = 0
zC = znode(6, grid, Center())
Δy = Ly/8
Δz = 500

#Δc = 2Δy
Δb = Ly/Ny * M²
ϵb = 1e-2 * Δb # noise amplitude

bᵢ(y, z) = N² * z - M² * Ly*sin(pi*y/Ly) *exp(-(3*z/Lz)^2)
cᵢ(y, z) = exp( -((y-yC)/Δy)^2 )*exp( -((z-zC)/Δz).^2);

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model; Δt, stop_time) #, stop_iteration = 10)

wall_clock = Ref(time_ns())

function progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[] = time_ns()
    
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(save_fields_interval),
                                                      filename = filename * "_fields",
                                                      overwrite_existing = true)

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

#####
##### Visualize
#####

fig = Figure(size=(1400, 700))

filepath = filename * "_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes(bt)

zscale = 1
z = z .* zscale

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(mean(ut[n], dims=1), 1, :, :)
bn(n) = interior(mean(bt[n], dims=1), 1, :, :)
cn(n) = interior(mean(ct[n], dims=1), 1, :, :)

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

axu = Axis(fig[2, 1], xlabel="$gradient (km)", ylabel="z (km)", title="Zonal velocity")
axc = Axis(fig[3, 1], xlabel="$gradient (km)", ylabel="z (km)", title="Tracer concentration")
slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
n = slider.value

u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)

hm = heatmap!(axu, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(axu, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

hm = heatmap!(axc, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:speed)
contour!(axc, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Baroclinic adjustment with GM at t = " * prettytime(times[$n])
ax_t = fig[1, 1:2] = Label(fig, title_str)

display(fig)

record(fig, filename * ".mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

