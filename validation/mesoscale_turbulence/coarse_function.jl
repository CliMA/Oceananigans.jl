using Printf
using Statistics
using Random
using Oceananigans
using Oceananigans.Units
using GLMakie



function baroclinic_adjustment(kskew, ksymmetric, tapering, scale; dt = 20minutes, gradient = "y")

# Architecture
architecture = CPU()

# Domain
Lx = 1000kilometers
Ly = 1000kilometers  # north-south extent [m]
Lz = 1kilometers     # depth [m]
Nx = 20
Ny = 20
Nz = 20
save_fields_interval = 0.5day
stop_time = 30days
Δt = dt

zfaces = -Lz .+ (-0.5 * (cos.(range(0.0, π, length=Nz + 1))) .+ 0.5) .* Lz # range(-Lz, 0.0, length = Nz+1)
zfaces = range(-Lz, 0.0, length=Nz + 1)

grid = RectilinearGrid(architecture;
                       topology = (Bounded, Bounded, Bounded),
                       size = (Nx, Ny, Nz), 
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       z = zfaces,
                       halo = (3, 3, 3))

coriolis = FPlane(latitude = -45)

println("The diffusive timescale is ", (zfaces[1] - zfaces[2])^2 / 1e3)

Δy = Ly/Ny
@show κh = νh = Δy^4 / 10days
vertical_closure = VerticalScalarDiffusivity(ν=1e-2, κ=1e-4)
horizontal_closure = HorizontalScalarBiharmonicDiffusivity(ν=νh, κ=κh)

gerdes_koberle_willebrand_tapering = FluxTapering(tapering)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew=kskew,
    κ_symmetric=ksymmetric,
    slope_limiter=gerdes_koberle_willebrand_tapering)

closures = (vertical_closure, horizontal_closure, gent_mcwilliams_diffusivity)

@info "Building a model..."

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    coriolis = coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    closure = closures,
                                    tracers = (:b, :c),
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface())

@info "Built $model."

"""
Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

For example:

y < y₀           => ramp = 0
y₀ < y < y₀ + Δy => ramp = y / Δy
y > y₀ + Δy      => ramp = 1
"""
function ramp(x, y, Δ)
    gradient == "x" && return min(max(0, x / Δ + 1/2), 1)
    gradient == "y" && return min(max(0, y / Δ + 1/2), 1)
    gradient == "xy" && return 0.5 *( min(max(0, x / Δ + 1/2), 1) + min(max(0, y / Δ + 1/2), 1) )
end

# Parameters
N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

Δy = 100kilometers * 1.0
Δz = 100

Δc = 2Δy
Δb = Δy * M²
ϵb = 0e-2 * Δb # noise amplitude

bᵢ(x, y, z) = N² * z + scale * Δb * ramp(x, y, Δy)
cᵢ(x, y, z) = 00.25 # exp(-x^2 / 2Δc^2) * exp(-y^2 / 2Δc^2) * exp(-(z + Lz/4)^2 / 2Δz^2)

set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation building
#####

simulation = Simulation(model; Δt, stop_time)
wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()
    
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(50))

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)
println("done with gradient in ", gradient)

loss = extrema(interior(simulation.model.tracers.c)[:, :, :, end])
return loss, simulation
end

klist = [0.0, 1e2, 1e3, 2e3]
taperinglist = [0.0, 1e-2, 2e-2, 0.5e-2, 10.0]
scalelist = [0.5, 1.0, 2.0]
namelist = (; κskew = 1e3, κsymmetric = 1e3, tapering = 1e-2, scale = 2.0)

clist = []
plist = []
for k in klist, tapering in taperinglist, scale in scalelist
    namelist = (; κskew=k, κsymmetric=k, tapering=tapering, scale=scale)
    println("current on ", namelist)
    cextrema, simulation = baroclinic_adjustment(namelist...)
    push!(clist, cextrema)
    push!(plist, namelist)
end

for i in eachindex(clist)
    if (abs(clist[i]) > 0.3) | (abs(clist[i]) < 0.2)
        println("-----------------------")
        println("For parameter values ", plist[i])
        println("the simulation was not good")
        println("----------------------")
    end
end

# (; κskew = 2000.0, κsymmetric = 2000.0, tapering = 0.01, scale = 2.0) failed