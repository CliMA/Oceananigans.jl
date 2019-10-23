using Oceananigans.TurbulenceClosures: VerstappenAnisotropicMinimumDissipation

interiordata(a, grid) = view(a, grid.Hx+1:grid.Nx+grid.Hx,
                                grid.Hy+1:grid.Ny+grid.Hy,
                                grid.Hz+1:grid.Nz+grid.Hz,
                            )

# Temporary until k index is reversed
const Nx, Ny, Nz = 16, 16, 16
const Hx, Hy, Hz = 1, 1, 1

function get_fields_from_checkpoint(filename)

    file = jldopen(filename)

    solution = (
        u = reverse(file["velocities/u"]; dims=3),
        v = reverse(file["velocities/v"]; dims=3),
        w = cat(zeros(Nx+2Hx, Ny+2Hy), reverse(file["velocities/w"][:, :, 2:Nz+2Hz]; dims=3); dims=3),
        T = reverse(file["tracers/T"]; dims=3),
        S = reverse(file["tracers/S"]; dims=3)
    )

    Gⁿ = (
        u = reverse(file["timestepper/Gⁿ/Gu"]; dims=3),
        v = reverse(file["timestepper/Gⁿ/Gv"]; dims=3),
        w = cat(zeros(Nx+2Hx, Ny+2Hy), reverse(file["timestepper/Gⁿ/Gw"][:, :, 2:Nz+2Hz]; dims=3); dims=3),
        T = reverse(file["timestepper/Gⁿ/GT"]; dims=3),
        S = reverse(file["timestepper/Gⁿ/GS"]; dims=3)
    )

    G⁻ = (
        u = reverse(file["timestepper/G⁻/Gu"]; dims=3),
        v = reverse(file["timestepper/G⁻/Gv"]; dims=3),
        w = cat(zeros(Nx+2Hx, Ny+2Hy), reverse(file["timestepper/G⁻/Gw"][:, :, 2:Nz+2Hz]; dims=3); dims=3),
        T = reverse(file["timestepper/G⁻/GT"]; dims=3),
        S = reverse(file["timestepper/G⁻/GS"]; dims=3)
    )

    close(file)

    return solution, Gⁿ, G⁻
end

function run_ocean_large_eddy_simulation_regression_test(arch, closure)
    name = "ocean_large_eddy_simulation_" * string(typeof(closure).name.wrapper)

    spinup_steps = 10000
      test_steps = 10
              Δt = 2.0

    # Parameters
      Qᵀ = 5e-5     # Temperature flux at surface
      Qᵘ = -2e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient

    # Boundary conditions
    u_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Qᵘ)       )
    T_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Qᵀ),
                                     bottom = BoundaryCondition(Gradient, ∂T∂z) )
    S_bcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, 5e-8)     )

    # Model instantiation
    model = Model(
             architecture = arch,
                     grid = RegularCartesianGrid(size=(16, 16, 16), length=(16, 16, 16)),
                 coriolis = FPlane(f=1e-4),
                 buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4, β=8e-4)),
                  closure = closure,
      boundary_conditions = BoundaryConditions(u=u_bcs, T=T_bcs, S=S_bcs)
    )

    ArrayType = typeof(model.velocities.u.data.parent)  # The type of the underlying data, not the offset array.

    #=
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
    pop!(model.output_writers, :checkpointer)
    =#

    initial_filename = joinpath(dirname(@__FILE__), "data", name * "_$spinup_steps.jld2")

    solution₀, Gⁿ₀, G⁻₀ = get_fields_from_checkpoint(initial_filename)

    model.velocities.u.data.parent .= ArrayType(solution₀.u)
    model.velocities.v.data.parent .= ArrayType(solution₀.v)
    model.velocities.w.data.parent .= ArrayType(solution₀.w)
    model.tracers.T.data.parent    .= ArrayType(solution₀.T)
    model.tracers.S.data.parent    .= ArrayType(solution₀.S)

    model.timestepper.Gⁿ.u.data.parent .= ArrayType(Gⁿ₀.u)
    model.timestepper.Gⁿ.v.data.parent .= ArrayType(Gⁿ₀.v)
    model.timestepper.Gⁿ.w.data.parent .= ArrayType(Gⁿ₀.w)
    model.timestepper.Gⁿ.T.data.parent .= ArrayType(Gⁿ₀.T)
    model.timestepper.Gⁿ.S.data.parent .= ArrayType(Gⁿ₀.S)

    model.timestepper.G⁻.u.data.parent .= ArrayType(G⁻₀.u)
    model.timestepper.G⁻.v.data.parent .= ArrayType(G⁻₀.v)
    model.timestepper.G⁻.w.data.parent .= ArrayType(G⁻₀.w)
    model.timestepper.G⁻.T.data.parent .= ArrayType(G⁻₀.T)
    model.timestepper.G⁻.S.data.parent .= ArrayType(G⁻₀.S)

    model.clock.time = spinup_steps * Δt
    model.clock.iteration = spinup_steps

    time_step!(model, test_steps, Δt; init_with_euler=false)

    final_filename = joinpath(dirname(@__FILE__), "data", name * "_$(spinup_steps+test_steps).jld2")

    solution₁, Gⁿ₁, G⁻₁ = get_fields_from_checkpoint(final_filename)

    for name in (:u, :v, :w, :T, :S)
        if name ∈ (:u, :v, :w)
            test_field = getproperty(model.velocities, name)
        else
            test_field = getproperty(model.tracers, name)
        end

        correct_field = getproperty(solution₁, name)

        Δ = Array(test_field.data.parent) .- correct_field

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g\n",
                       name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std))
    end

    @test all(Array(interiordata(solution₁.u, model.grid)) .≈ Array(data(model.velocities.u)))
    @test all(Array(interiordata(solution₁.v, model.grid)) .≈ Array(data(model.velocities.v)))
    @test all(Array(interiordata(solution₁.w, model.grid)) .≈ Array(data(model.velocities.w)))
    @test all(Array(interiordata(solution₁.T, model.grid)) .≈ Array(data(model.tracers.T)))
    @test all(Array(interiordata(solution₁.S, model.grid)) .≈ Array(data(model.tracers.S)))

    return nothing
end
