function run_ocean_large_eddy_simulation_regression_test(arch, closure)
    name = "ocean_large_eddy_simulation_" * string(typeof(closure).name.wrapper)

    spinup_steps = 10000
      test_steps = 10
              Δt = 2.0

    #=
    # Parameters
      Qᵀ = 5e-5     # Temperature flux at surface
      Qᵘ = -2e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient

    # Boundary conditions
    u_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Qᵘ)       )
    T_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Qᵀ),
                                     bottom = BoundaryCondition(Gradient, ∂T∂z) )
    S_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, 5e-8)   )

    # Model instantiation
    model = Model(
             architecture = arch,  
                     grid = RegularCartesianGrid(N=(16, 16, 16), L=(16, 16, 16)), 
                 coriolis = FPlane(f=1e-4),
                 buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=8e-4)),
                  closure = closure,
      boundary_conditions = BoundaryConditions(u=u_bcs, T=T_bcs, S=S_bcs)
    )

    # Initialize model: random noise damped at top and bottom
    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
    T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-2 * Ξ(z)
    u₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=u₀, w=u₀, T=T₀, S=35)

    time_step!(model, spinup_steps-test_steps, Δt)

    model.output_writers[:checkpointer] = Checkpointer(model;
                                                       frequency = test_steps,
                                                          prefix = name * "_",
                                                             dir = joinpath(dirname(@__FILE__), "data")
                                                      )
                                                       
    time_step!(model, 2test_steps, Δt)
    =#

    test_model = restore_from_checkpoint(joinpath(dirname(@__FILE__), "data", name * "_$spinup_steps.jld2"))
                                         
    time_step!(test_model, test_steps, Δt; init_with_euler=false)

    checkpointed_model = restore_from_checkpoint(joinpath(dirname(@__FILE__), "data", name * "_$(spinup_steps+test_steps).jld2"))

    @test all(Array(data(checkpointed_model.velocities.u)) .≈ Array(data(test_model.velocities.u)))
    @test all(Array(data(checkpointed_model.velocities.v)) .≈ Array(data(test_model.velocities.v)))
    @test all(Array(data(checkpointed_model.velocities.w)) .≈ Array(data(test_model.velocities.w)))
    @test all(Array(data(checkpointed_model.tracers.T))    .≈ Array(data(test_model.tracers.T)))
    @test all(Array(data(checkpointed_model.tracers.S))    .≈ Array(data(test_model.tracers.S)))

    return nothing
end
