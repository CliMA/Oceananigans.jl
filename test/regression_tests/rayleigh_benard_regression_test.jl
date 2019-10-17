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
    S★(x, z) = exp(4z) * sin(2π/Lx * x)
    FS(i, j, k, grid, time, U, Φ, params) = 1/10 * (S★(grid.xC[i], grid.zC[k]) - Φ.S[i, j, k])

    model = Model(
               architecture = arch,
                       grid = RegularCartesianGrid(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz)),
                    closure = ConstantIsotropicDiffusivity(ν=ν, κ=κ),
                   buoyancy = BuoyancyTracer(), 
        boundary_conditions = BoundaryConditions(T=HorizontallyPeriodicBCs(
                                top=BoundaryCondition(Value, 0.0), bottom=BoundaryCondition(Value, Δb))),
                    forcing = ModelForcing(S=FS)
    )

    ArrayType = typeof(model.velocities.u.data.parent)  # The type of the underlying data, not the offset array.
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
    set_ic!(model, T=b₀)

    time_step!(model, spinup_steps-test_steps, Δt)
    push!(model.output_writers, outputwriter)
    time_step!(model, 2test_steps, Δt)
    =#

    #####
    ##### Regression test
    #####

    # Load initial state
    u₀, v₀, w₀ = get_output_tuple(outputwriter, spinup_steps, :U)
    T₀, S₀ = get_output_tuple(outputwriter, spinup_steps, :Φ)
    Gu, Gv, Gw, GT, GS = get_output_tuple(outputwriter, spinup_steps, :G)

    data(model.velocities.u) .= ArrayType(u₀)
    data(model.velocities.v) .= ArrayType(v₀)
    data(model.velocities.w) .= ArrayType(w₀)
    data(model.tracers.T)    .= ArrayType(T₀)
    data(model.tracers.S)    .= ArrayType(S₀)

    data(model.timestepper.Gⁿ.Gu) .= ArrayType(Gu)
    data(model.timestepper.Gⁿ.Gv) .= ArrayType(Gv)
    data(model.timestepper.Gⁿ.Gw) .= ArrayType(Gw)
    data(model.timestepper.Gⁿ.GT) .= ArrayType(GT)
    data(model.timestepper.Gⁿ.GS) .= ArrayType(GS)

    model.clock.iteration = spinup_steps
    model.clock.time = spinup_steps * Δt
    length(model.output_writers) > 0 && pop!(model.output_writers)

    # Step the model forward and perform the regression test
    time_step!(model, test_steps, Δt; init_with_euler=false)

    u₁, v₁, w₁ = get_output_tuple(outputwriter, spinup_steps+test_steps, :U)
    T₁, S₁ = get_output_tuple(outputwriter, spinup_steps+test_steps, :Φ)

    field_names = ["u", "v", "w", "T", "S"]
    fields = [model.velocities.u, model.velocities.v, model.velocities.w, model.tracers.T, model.tracers.S]
    fields_gm = [u₁, v₁, w₁, T₁, S₁]
    for (field_name, φ, φ_gm) in zip(field_names, fields, fields_gm)
        φ_min = minimum(Array(data(φ)) - φ_gm)
        φ_max = maximum(Array(data(φ)) - φ_gm)
        φ_mean = mean(Array(data(φ)) - φ_gm)
        φ_abs_mean = mean(abs.(Array(data(φ)) - φ_gm))
        φ_std = std(Array(data(φ)) - φ_gm)
        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       field_name, φ_min, φ_max, φ_mean, φ_abs_mean, φ_std))
    end

    # Now test that the model state matches the regression output.
    @test all(Array(data(model.velocities.u)) .≈ u₁)
    @test all(Array(data(model.velocities.v)) .≈ v₁)
    @test all(Array(data(model.velocities.w)) .≈ w₁)
    @test all(Array(data(model.tracers.T))    .≈ T₁)
    @test all(Array(data(model.tracers.S))    .≈ S₁)
    return nothing
end

