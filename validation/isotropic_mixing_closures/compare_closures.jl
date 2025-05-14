using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: Smagorinsky, DynamicCoefficient, LagrangianAveraging
using Printf

stopwatch = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - stopwatch[])

    msg = @sprintf("Iter: %d, time: %s, Δt: %s, wall time: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed))

    u, v, w = sim.model.velocities
    msg *= @sprintf(", max|u|: (%.4f, %.4f, %.4f) m s⁻¹",
                    maximum(abs, interior(u)),
                    maximum(abs, interior(v)),
                    maximum(abs, interior(w)))

    @info msg

    stopwatch[] = time_ns()

    return nothing
end

function wind_driven_turbulence_simulation(grid, advection, closure; stop_time=9hours, τx=-1e-4, f=1e-4, N²=1e-5)
    @info "Running closure $closure"
    coriolis = FPlane(; f)
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx))
    model = NonhydrostaticModel(; grid, closure, coriolis, advection,
                                boundary_conditions = (; u=u_bcs),
                                tracers = :b,
                                buoyancy = BuoyancyTracer())

    Δz = minimum_zspacing(grid)
    δb = N² * Δz
    u★ = sqrt(abs(τx))
    uᵢ(x, y, z) = 1e-2 * u★ * (2rand() - 1)
    bᵢ(x, y, z) = N² * z + 1e-2 * δb
    set!(model, u=uᵢ, v=uᵢ, w=uᵢ,  b=bᵢ)

    Δt = 1e-1 * Δz / u★
    simulation = Simulation(model; Δt, stop_time)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    add_callback!(simulation, progress, IterationInterval(100))

    return simulation
end

arch = GPU()
Nx = Ny = Nz = 128
x = y = (0, 128)
z = (-64, 0)
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(5, 5, 5), x, y, z, topology=(Periodic, Periodic, Bounded))
Δz = @show 10 * round(Int, - z[1] / Nz)
save_interval = 1hour

schedule = TimeInterval(save_interval)
filename = "wind_driven_WENO9_$Δz"
advection = WENO(order=9)
closure = nothing
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_WENO5_$Δz"
advection = WENO(order=5)
closure = nothing
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_AMD_$Δz"
advection = Centered(order=2)
closure = AnisotropicMinimumDissipation()
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
νₑ = simulation.model.diffusivity_fields.νₑ
κₑ = simulation.model.diffusivity_fields.κₑ.b
outputs = merge(outputs, (; νₑ, κₑ))
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_smagorinsky_lilly_$Δz"
advection = Centered(order=2)
closure = SmagorinskyLilly()
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
νₑ = simulation.model.diffusivity_fields.νₑ
outputs = merge(outputs, (; νₑ))
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_constant_smagorinsky_$Δz"
advection = Centered(order=2)
closure = Smagorinsky(coefficient=0.16)
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
νₑ = simulation.model.diffusivity_fields.νₑ
outputs = merge(outputs, (; νₑ))
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_directional_smagorinsky_$Δz"
advection = Centered(order=2)
closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2)))
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
𝒥ᴸᴹ = simulation.model.diffusivity_fields.𝒥ᴸᴹ
𝒥ᴹᴹ = simulation.model.diffusivity_fields.𝒥ᴹᴹ
νₑ = simulation.model.diffusivity_fields.νₑ
outputs = merge(outputs, (; 𝒥ᴸᴹ, 𝒥ᴹᴹ, νₑ))
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(save_interval)
filename = "wind_driven_lagrangian_smagorinsky_$Δz"
advection = Centered(order=2)
closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=LagrangianAveraging()))
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
𝒥ᴸᴹ = simulation.model.diffusivity_fields.𝒥ᴸᴹ
𝒥ᴹᴹ = simulation.model.diffusivity_fields.𝒥ᴹᴹ
𝒥ᴸᴹ⁻ = simulation.model.diffusivity_fields.𝒥ᴸᴹ⁻
𝒥ᴹᴹ⁻ = simulation.model.diffusivity_fields.𝒥ᴹᴹ⁻
νₑ = simulation.model.diffusivity_fields.νₑ
outputs = merge(outputs, (; 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, νₑ))
output_writer = JLD2Writer(simulation.model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)
