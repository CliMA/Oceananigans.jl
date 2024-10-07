using Oceananigans.Grids: xnode, znode
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.DistributedComputations: cpu_architecture, partition, reconstruct_global_grid

function run_rayleigh_benard_regression_test(arch, grid_type)

    #####
    ##### Parameters
    #####
          α = 2                 # aspect ratio
          n = 1                 # resolution multiple
         Ra = 1e6               # Rayleigh number
    Nx = Ny = 8n * α            # horizontal resolution
    Lx = Ly = 1.0 * α           # horizontal extent
         Nz = 16n               # vertical resolution
         Lz = 1.0               # vertical extent
         Pr = 0.7               # Prandtl number
          a = 1e-1              # noise amplitude for initial condition
         Δb = 1.0               # buoyancy differential

    # Rayleigh and Prandtl determine transport coefficients
    ν = sqrt(Δb * Pr * Lz^3 / Ra)
    κ = ν / Pr

    #####
    ##### Model setup
    #####

    if grid_type == :regular
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), halo=(1, 1, 1))
    elseif grid_type == :vertically_unstretched
        zF = range(-Lz, 0, length=Nz+1)
        grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=zF, halo=(1, 1, 1))
    end

    # Force salinity as a passive tracer (βS=0)
    c★(x, z) = exp(4z) * sin(2π/Lx * x)

    function Fc(i, j, k, grid, clock, model_fields)
        x = xnode(i, grid, Center())
        z = znode(k, grid, Center())
        return 1/10 * (c★(x, z) - model_fields.c[i, j, k])
    end

    cforcing = Forcing(Fc, discrete_form=true)

    bbcs = FieldBoundaryConditions(top = BoundaryCondition(Value(), 0.0),
                                   bottom = BoundaryCondition(Value(), Δb))

    model = NonhydrostaticModel(; grid,
                                timestepper = :QuasiAdamsBashforth2,
                                closure = ScalarDiffusivity(ν=ν, κ=κ),
                                tracers = (:b, :c),
                                buoyancy = Buoyancy(model=BuoyancyTracer()),
                                boundary_conditions = (; b=bbcs),
                                hydrostatic_pressure_anomaly = CenterField(grid),
                                forcing = (; c=cforcing))

    # Lz/Nz will work for both the :regular and :vertically_unstretched grids.
    Δt = 0.01 * min(model.grid.Δxᶜᵃᵃ, model.grid.Δyᵃᶜᵃ, Lz/Nz)^2 / ν

    # We will manually change the stop_iteration as needed.
    simulation = Simulation(model, Δt=Δt, stop_iteration=0)

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)

    spinup_steps = 1000
      test_steps = 100

    prefix = "rayleigh_benard"

    checkpointer = Checkpointer(model, schedule=IterationInterval(test_steps), prefix=prefix,
                                dir=joinpath(dirname(@__FILE__), "data"))

    u, v, w = model.velocities
    b, c = model.tracers

    #####
    ##### Initial condition and spinup steps for creating regression test data
    #####

    #=
    @warn "Generating new data for the Rayleigh-Benard regression test."

    ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
    b₀(x, y, z) = (ξ(z) - z) / Lz
    set!(model, b=b₀)

    simulation.stop_iteration = spinup_steps-test_steps
    run!(simulation)

    push!(simulation.output_writers, checkpointer)
    simulation.stop_iteration += 2test_steps
    run!(simulation)
    =#

    #####
    ##### Regression test
    #####

    # Load initial state
    datadep_path = "regression_test_data/" * prefix * "_iteration$spinup_steps.jld2"
    initial_filename = @datadep_str datadep_path

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    cpu_arch = cpu_architecture(architecture(grid))

    u₀ = partition(ArrayType(solution₀.u[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(u))
    v₀ = partition(ArrayType(solution₀.v[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(v))
    w₀ = partition(ArrayType(solution₀.w[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(w))
    b₀ = partition(ArrayType(solution₀.b[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(b))
    c₀ = partition(ArrayType(solution₀.c[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(c))

    Gⁿu₀ = partition(ArrayType(Gⁿ₀.u[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(u))
    Gⁿv₀ = partition(ArrayType(Gⁿ₀.v[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(v))
    Gⁿw₀ = partition(ArrayType(Gⁿ₀.w[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(w))
    Gⁿb₀ = partition(ArrayType(Gⁿ₀.b[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(b))
    Gⁿc₀ = partition(ArrayType(Gⁿ₀.c[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(c))

    G⁻u₀ = partition(ArrayType(G⁻₀.u[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(u))
    G⁻v₀ = partition(ArrayType(G⁻₀.v[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(v))
    G⁻w₀ = partition(ArrayType(G⁻₀.w[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(w))
    G⁻b₀ = partition(ArrayType(G⁻₀.b[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(b))
    G⁻c₀ = partition(ArrayType(G⁻₀.c[2:end-1, 2:end-1, 2:end-1]), cpu_arch, size(c))

    set!(model, u = u₀, v = v₀, w = w₀, b = b₀, c = c₀)

    set!(model.timestepper.Gⁿ.u, Gⁿu₀)
    set!(model.timestepper.Gⁿ.v, Gⁿv₀)
    set!(model.timestepper.Gⁿ.w, Gⁿw₀)
    set!(model.timestepper.Gⁿ.b, Gⁿb₀)
    set!(model.timestepper.Gⁿ.c, Gⁿc₀)

    set!(model.timestepper.G⁻.u, G⁻u₀)
    set!(model.timestepper.G⁻.v, G⁻v₀)
    set!(model.timestepper.G⁻.w, G⁻w₀)
    set!(model.timestepper.G⁻.b, G⁻b₀)
    set!(model.timestepper.G⁻.c, G⁻c₀)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(simulation.output_writers) > 0 && pop!(simulation.output_writers)

    # Step the model forward and perform the regression test
    update_state!(model)

    model.clock.last_Δt = Δt

    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    datadep_path = "regression_test_data/" * prefix * "_iteration$(spinup_steps+test_steps).jld2"
    final_filename = @datadep_str datadep_path

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    test_fields =  CUDA.@allowscalar (u = Array(interior(model.velocities.u)),
                                      v = Array(interior(model.velocities.v)),
                                      w = Array(interior(model.velocities.w)[:, :, 1:Nz]),
                                      b = Array(interior(model.tracers.b)),
                                      c = Array(interior(model.tracers.c)))

    global_grid = reconstruct_global_grid(model.grid)

    u₁ = interior(solution₁.u, global_grid)
    v₁ = interior(solution₁.v, global_grid)
    w₁ = interior(solution₁.w, global_grid)
    b₁ = interior(solution₁.b, global_grid)
    c₁ = interior(solution₁.c, global_grid)

    reference_fields = (u = partition(Array(u₁), cpu_arch, size(u)),
                        v = partition(Array(v₁), cpu_arch, size(v)),
                        w = partition(Array(w₁), cpu_arch, size(test_fields.w)),
                        b = partition(Array(b₁), cpu_arch, size(b)),
                        c = partition(Array(c₁), cpu_arch, size(c)))

    summarize_regression_test(test_fields, reference_fields)

    CUDA.allowscalar(true)
    @test all(test_fields.u .≈ reference_fields.u)
    @test all(test_fields.v .≈ reference_fields.v)
    @test all(test_fields.w .≈ reference_fields.w)
    @test all(test_fields.b .≈ reference_fields.b)
    @test all(test_fields.c .≈ reference_fields.c)
    CUDA.allowscalar(false)

    return nothing
end
