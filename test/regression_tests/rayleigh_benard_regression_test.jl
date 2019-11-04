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

    # Force salinity as a passive tracer (βS=0)
    c★(x, z) = exp(4z) * sin(2π/Lx * x)
    Fc(i, j, k, grid, time, U, C, params) = 1/10 * (c★(grid.xC[i], grid.zC[k]) - C.c[i, j, k])

    bbcs = HorizontallyPeriodicBCs(   top = BoundaryCondition(Value, 0.0),
                                   bottom = BoundaryCondition(Value, Δb))

    model = Model(
               architecture = arch,
                       grid = RegularCartesianGrid(size=(Nx, Ny, Nz), length=(Lx, Ly, Lz)),
                    closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
                    tracers = (:b, :c),
                   buoyancy = BuoyancyTracer(),
        boundary_conditions = BoundaryConditions(b=bbcs),
                    forcing = ModelForcing(c=Fc)
    )

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(model.velocities.u.data.parent)

    Δt = 0.01 * min(model.grid.Δx, model.grid.Δy, model.grid.Δz)^2 / ν

    spinup_steps = 1000
      test_steps = 100

    output_U(model) = datatuple(model.velocities)
    output_Φ(model) = datatuple(model.tracers)
    output_G(model) = datatuple(model.timestepper.Gⁿ)
    outputfields = Dict(:U=>output_U, :Φ=>output_Φ, :G=>output_G)

    prefix = "data_rayleigh_benard_regression"
    outputwriter = JLD2OutputWriter(model, outputfields; dir=joinpath(dirname(@__FILE__), "data"),
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
    set_ic!(model, b=b₀)

    time_step!(model, spinup_steps-test_steps, Δt)
    push!(model.output_writers, outputwriter)
    time_step!(model, 2test_steps, Δt)
    =#


    #####
    ##### Regression test
    #####

    # Load initial state
    u₀, v₀, w₀ = get_output_tuple(outputwriter, spinup_steps, :U)
    b₀, c₀ = get_output_tuple(outputwriter, spinup_steps, :Φ)
    Gu₀, Gv₀, Gw₀, Gb₀, Gc₀ = get_output_tuple(outputwriter, spinup_steps, :G)

    model.velocities.u.data.parent .= ArrayType(u₀.parent)
    model.velocities.v.data.parent .= ArrayType(v₀.parent)
    model.velocities.w.data.parent .= ArrayType(w₀.parent)
    model.tracers.b.data.parent    .= ArrayType(b₀.parent)
    model.tracers.c.data.parent    .= ArrayType(c₀.parent)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gu₀.parent)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gv₀.parent)
    model.timestepper.Gⁿ.w.data.parent .= ArrayType(Gw₀.parent)
    model.timestepper.Gⁿ.b.data.parent .= ArrayType(Gb₀.parent)
    model.timestepper.Gⁿ.c.data.parent .= ArrayType(Gc₀.parent)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(model.output_writers) > 0 && pop!(model.output_writers)

    # Step the model forward and perform the regression test
    time_step!(model, test_steps, Δt; init_with_euler=false)

    u₁, v₁, w₁ = get_output_tuple(outputwriter, spinup_steps+test_steps, :U)
    b₁, c₁ = get_output_tuple(outputwriter, spinup_steps+test_steps, :Φ)

    field_names = ["u", "v", "w", "b", "c"]
    fields = [model.velocities.u.data.parent, model.velocities.v.data.parent,
              model.velocities.w.data.parent, model.tracers.b.data.parent,
              model.tracers.c.data.parent]
    fields_correct = [u₁.parent, v₁.parent, w₁.parent,
                      b₁.parent, c₁.parent]
    summarize_regression_test(field_names, fields, fields_correct)

    # Now test that the model state matches the regression output.
    @test all(Array(model.velocities.u.data.parent) .≈ u₁.parent)
    @test all(Array(model.velocities.v.data.parent) .≈ v₁.parent)
    @test all(Array(model.velocities.w.data.parent) .≈ w₁.parent)
    @test all(Array(model.tracers.b.data.parent)    .≈ b₁.parent)
    @test all(Array(model.tracers.c.data.parent)    .≈ c₁.parent)

    return nothing
end
