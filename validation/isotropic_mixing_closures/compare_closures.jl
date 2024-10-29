using Oceananigans

function progress(sim)
    msg = @sprintf("Iter: %d, time: %s, Δt: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt))

    u, v, w = sim.model.velocities
    msg *= @sprintf(", max|u|: (%.4f, %.4f, %.4f) m s⁻¹",
                    maximum(abs, interior(u)),
                    maximum(abs, interior(v)),
                    maximum(abs, interior(w)))

    @info msg

    return nothing
end

function wind_driven_turbulence_simulation(grid, advection, closure; stop_time=12hours, τx=-1e-4, f=1e-4, N²=1e-5)
    coriolis = FPlane(; f)
    u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(τx))
    model = NonhydrostaticModel(; grid, closure, coriolis, boundary_conditions,
                                tracers=:b, buoyancy=BuoyancyTracer())

    Δz = minimum_zspacing(grid)
    δb = N² * Δz
    u★ = sqrt(abs(τx))
    uᵢ(x, y, z) = 1e-2 * u★ * (2rand() - 1)
    bᵢ(x, y, z) = N² * z + 1e-2 * δb
    set!(model, u=uᵢ, v=vᵢ, w=wᵢ,  b=bᵢ)

    Δt = 1e-1 * Δz / u★
    simulation = Simulation(model; Δt, stop_time)
    conjure_time_step_wizard!(simulation, cfl=0.5)
    add_callback!(simulation, progress, IterationInterval(10))

    return simulation
end

arch = GPU()
Nx = Ny = Nz = 64
x = y = (0, 128)
z = (-64, 0)
grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), x, y, z, topology=(Periodic, Periodic, Bounded))

schedule = TimeInterval(3hours)
filename = "wind_driven_WENO"
advection = WENO(order=9)
closure = nothing
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
output_writer = JLD2OutputWriter(model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(3hours)
filename = "wind_driven_AMD"
advection = Centered(order=2)
closure = AnisotropicMinimumDissipation()
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
output_writer = JLD2OutputWriter(model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(3hours)
filename = "wind_driven_constant_smagorinsky"
advection = Centered(order=2)
closure = Smagorinsky(coefficient=0.16)
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
output_writer = JLD2OutputWriter(model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

schedule = TimeInterval(3hours)
filename = "wind_driven_dynamic_smagorinsky"
advection = Centered(order=2)
closure = Smagorinsky(coefficient=DynamicCoefficient(averaging=(1, 2)))
simulation = wind_driven_turbulence_simulation(grid, advection, closure)
outputs = merge(simulation.model.velocities, simulation.model.tracers)
𝒥ᴸᴹ = simulation.model.diffusivity_fields.𝒥ᴸᴹ
𝒥ᴹᴹ = simulation.model.diffusivity_fields.𝒥ᴹᴹ
outputs = merge(outputs, (; 𝒥ᴸᴹ, 𝒥ᴹᴹ))
output_writer = JLD2OutputWriter(model, outputs; filename, schedule, overwrite_existing=true)
simulation.output_writers[:jld2] = output_writer
run!(simulation)

