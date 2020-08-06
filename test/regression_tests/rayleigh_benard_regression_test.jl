using Oceananigans.Grids: xnode, znode

function run_rayleigh_benard_regression_test(arch)

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

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz))

    # Force salinity as a passive tracer (βS=0)
    c★(x, z) = exp(4z) * sin(2π/Lx * x)
    Fc(i, j, k, grid, clock, state) = 1/10 * (c★(xnode(Cell, i, grid), znode(Cell, k, grid)) - state.tracers.c[i, j, k])

    bbcs = TracerBoundaryConditions(grid,    top = BoundaryCondition(Value, 0.0),
                                          bottom = BoundaryCondition(Value, Δb))

    model = IncompressibleModel(
               architecture = arch,
                       grid = grid,
                    closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
                    tracers = (:b, :c),
                   buoyancy = BuoyancyTracer(),
        boundary_conditions = (b=bbcs,),
                    forcing = ModelForcing(c=Fc)
    )

    Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

    # We will manually change the stop_iteration as needed.
    simulation = Simulation(model, Δt=Δt, stop_iteration=0)

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)

    spinup_steps = 1000
      test_steps = 100

    prefix = "rayleigh_benard"

    checkpointer = Checkpointer(model, iteration_interval=test_steps, prefix=prefix,
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
    initial_filename = joinpath(dirname(@__FILE__), "data", prefix * "_iteration$spinup_steps.jld2")

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    model.velocities.u.data.parent .= ArrayType(solution₀.u)
    model.velocities.v.data.parent .= ArrayType(solution₀.v)
    model.velocities.w.data.parent .= ArrayType(solution₀.w)
    model.tracers.b.data.parent    .= ArrayType(solution₀.b)
    model.tracers.c.data.parent    .= ArrayType(solution₀.c)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gⁿ₀.u)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gⁿ₀.v)
    model.timestepper.Gⁿ.w.data.parent .= ArrayType(Gⁿ₀.w)
    model.timestepper.Gⁿ.b.data.parent .= ArrayType(Gⁿ₀.b)
    model.timestepper.Gⁿ.c.data.parent .= ArrayType(Gⁿ₀.c)

    model.timestepper.G⁻.u.data.parent .= ArrayType(G⁻₀.u)
    model.timestepper.G⁻.v.data.parent .= ArrayType(G⁻₀.v)
    model.timestepper.G⁻.w.data.parent .= ArrayType(G⁻₀.w)
    model.timestepper.G⁻.b.data.parent .= ArrayType(G⁻₀.b)
    model.timestepper.G⁻.c.data.parent .= ArrayType(G⁻₀.c)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(simulation.output_writers) > 0 && pop!(simulation.output_writers)

    # Step the model forward and perform the regression test
    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    final_filename = joinpath(dirname(@__FILE__), "data", prefix * "_iteration$(spinup_steps+test_steps).jld2")

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    field_names = ["u", "v", "w", "b", "c"]

    test_fields = (model.velocities.u.data.parent,
                   model.velocities.v.data.parent,
                   model.velocities.w.data.parent,
                   model.tracers.b.data.parent,
                   model.tracers.c.data.parent)

    correct_fields = (solution₁.u,
                      solution₁.v,
                      solution₁.w,
                      solution₁.b,
                      solution₁.c)

    summarize_regression_test(field_names, test_fields, correct_fields)

    # Now test that the model state matches the regression output.
    @test all(Array(interior(solution₁.u, model.grid)) .≈ Array(interior(model.velocities.u)))
    @test all(Array(interior(solution₁.v, model.grid)) .≈ Array(interior(model.velocities.v)))
    @test all(Array(interior(solution₁.w, model.grid)) .≈ Array(interior(model.velocities.w)[:, :, 1:Nz]))
    @test all(Array(interior(solution₁.b, model.grid)) .≈ Array(interior(model.tracers.b)))
    @test all(Array(interior(solution₁.c, model.grid)) .≈ Array(interior(model.tracers.c)))

    return nothing
end
