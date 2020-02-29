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

    grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz))

    # Force salinity as a passive tracer (βS=0)
    c★(x, z) = exp(4z) * sin(2π/Lx * x)
    Fc(i, j, k, grid, time, U, C, params) = 1/10 * (c★(grid.xC[i], grid.zC[k]) - C.c[i, j, k])

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

    output_U(model) = datatuple(model.velocities)
    output_Φ(model) = datatuple(model.tracers)
    output_G(model) = datatuple(model.timestepper.Gⁿ)
    outputfields = Dict(:U=>output_U, :Φ=>output_Φ, :G=>output_G)

    prefix = "data_rayleigh_benard_regression"
    output_writer =
      JLD2OutputWriter(model, outputfields, dir=joinpath(dirname(@__FILE__), "data"),
                       prefix=prefix, frequency=test_steps, including=[])

    #####
    ##### Initial condition and spinup steps for creating regression test data
    #####

    #=
    @warn ("Generating new data for the Rayleigh-Benard regression test.
           New regression test data generation will fail unless the JLD2
           file $prefix.jld2 is manually deleted.")

    ξ(z) = a * rand() * z * (Lz + z) # noise, damped at the walls
    b₀(x, y, z) = (ξ(z) - z) / Lz
    set!(model, b=b₀)

    simulation.stop_iteration = spinup_steps-test_steps
    run!(simulation)

    push!(simulation.output_writers, output_writer)
    simulation.stop_iteration += 2test_steps
    run!(simulation)
    =#

    #####
    ##### Regression test
    #####

    # Load initial state
    u₀, v₀, w₀ = get_output_tuple(output_writer, spinup_steps, :U)
    b₀, c₀ = get_output_tuple(output_writer, spinup_steps, :Φ)
    Gu₀, Gv₀, Gw₀, Gb₀, Gc₀ = get_output_tuple(output_writer, spinup_steps, :G)

    model.velocities.u.data.parent .= ArrayType(u₀.parent)
    model.velocities.v.data.parent .= ArrayType(v₀.parent)
    model.velocities.w.data.parent[:, :, 1:Nz+2] .= ArrayType(w₀.parent)
    model.tracers.b.data.parent    .= ArrayType(b₀.parent)
    model.tracers.c.data.parent    .= ArrayType(c₀.parent)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gu₀.parent)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gv₀.parent)
    model.timestepper.Gⁿ.w.data.parent[:, :, 1:Nz+2] .= ArrayType(Gw₀.parent)
    model.timestepper.Gⁿ.b.data.parent .= ArrayType(Gb₀.parent)
    model.timestepper.Gⁿ.c.data.parent .= ArrayType(Gc₀.parent)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(simulation.output_writers) > 0 && pop!(Simulation.output_writers)

    # Step the model forward and perform the regression test
    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    u₁, v₁, w₁ = get_output_tuple(output_writer, spinup_steps+test_steps, :U)
    b₁, c₁ = get_output_tuple(output_writer, spinup_steps+test_steps, :Φ)

    field_names = ["u", "v", "w", "b", "c"]

    test_fields = (model.velocities.u.data.parent, 
                   model.velocities.v.data.parent,
                   model.velocities.w.data.parent[:, :, 1:Nz+2], 
                   model.tracers.b.data.parent,
                   model.tracers.c.data.parent)

    correct_fields = (u₁.parent, 
                      v₁.parent, 
                      w₁.parent,
                      b₁.parent, 
                      c₁.parent)

    summarize_regression_test(field_names, test_fields, correct_fields)

    # Now test that the model state matches the regression output.
    @test all(Array(model.velocities.u.data.parent) .≈ u₁.parent)
    @test all(Array(model.velocities.v.data.parent) .≈ v₁.parent)
    @test all(Array(model.velocities.w.data.parent)[:, :, 1:Nz+2] .≈ w₁.parent)
    @test all(Array(model.tracers.b.data.parent)    .≈ b₁.parent)
    @test all(Array(model.tracers.c.data.parent)    .≈ c₁.parent)

    return nothing
end
