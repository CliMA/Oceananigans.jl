using Oceananigans.Grids: xnode, znode
using Oceananigans.TimeSteppers: update_state!

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

    bbcs = FieldBoundaryConditions(top = BoundaryCondition(Value, 0.0),
                                   bottom = BoundaryCondition(Value, Δb))

    model = NonhydrostaticModel(; grid,
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

    u₀ = partition_global_array(cpu_arch, ArrayType(solution₀.u), size(solution₀.u))
    v₀ = partition_global_array(cpu_arch, ArrayType(solution₀.v), size(solution₀.v))
    w₀ = partition_global_array(cpu_arch, ArrayType(solution₀.w), size(solution₀.w))
    b₀ = partition_global_array(cpu_arch, ArrayType(solution₀.b), size(solution₀.b))
    c₀ = partition_global_array(cpu_arch, ArrayType(solution₀.c), size(solution₀.c))

    Gⁿu₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.u), size(Gⁿ₀.u))
    Gⁿv₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.v), size(Gⁿ₀.v))
    Gⁿw₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.w), size(Gⁿ₀.w))
    Gⁿb₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.b), size(Gⁿ₀.b))
    Gⁿc₀ = partition_global_array(cpu_arch, ArrayType(Gⁿ₀.c), size(Gⁿ₀.c))

    G⁻u₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.u), size(G⁻₀.u))
    G⁻v₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.v), size(G⁻₀.v))
    G⁻w₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.w), size(G⁻₀.w))
    G⁻b₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.b), size(G⁻₀.b))
    G⁻c₀ = partition_global_array(cpu_arch, ArrayType(G⁻₀.c), size(G⁻₀.c))

    model.velocities.u.data.parent .= u₀
    model.velocities.v.data.parent .= v₀
    model.velocities.w.data.parent .= w₀
    model.tracers.b.data.parent    .= b₀
    model.tracers.c.data.parent    .= c₀

    model.timestepper.Gⁿ.u.data.parent .= Gⁿu₀
    model.timestepper.Gⁿ.v.data.parent .= Gⁿv₀
    model.timestepper.Gⁿ.w.data.parent .= Gⁿw₀
    model.timestepper.Gⁿ.b.data.parent .= Gⁿb₀
    model.timestepper.Gⁿ.c.data.parent .= Gⁿc₀

    model.timestepper.G⁻.u.data.parent .= G⁻u₀
    model.timestepper.G⁻.v.data.parent .= G⁻v₀
    model.timestepper.G⁻.w.data.parent .= G⁻w₀
    model.timestepper.G⁻.b.data.parent .= G⁻b₀
    model.timestepper.G⁻.c.data.parent .= G⁻c₀

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

    u₁ = partition_global_array(cpu_arch, ArrayType(interior(solution₁.u)), size(solution₁.u))
    v₁ = partition_global_array(cpu_arch, ArrayType(interior(solution₁.v)), size(solution₁.v))
    w₁ = partition_global_array(cpu_arch, ArrayType(interior(solution₁.w)), size(solution₁.w))
    b₁ = partition_global_array(cpu_arch, ArrayType(interior(solution₁.b)), size(solution₁.b))
    c₁ = partition_global_array(cpu_arch, ArrayType(interior(solution₁.c)), size(solution₁.c))

    correct_fields = (u = u₁,
                      v = v₁,
                      w = w₁,
                      b = b₁,
                      c = c₁)

    summarize_regression_test(test_fields, correct_fields)

    CUDA.allowscalar(true)
    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.b .≈ correct_fields.b)
    @test all(test_fields.c .≈ correct_fields.c)
    CUDA.allowscalar(false)

    return nothing
end
