using Oceananigans.TurbulenceClosures: AnisotropicMinimumDissipation, LagrangianAveraging, initialize_closure_fields!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.DistributedComputations: cpu_architecture, partition

# Extract a window matching `target_size` from the center of `data`,
# auto-detecting and stripping halo layers. Works for any saved halo
# count (e.g. data saved as `parent(field)` with halo=1 is 18³,
# halo=2 is 20³; both reduce to the same 16³ interior). Also unwraps
# OffsetArrays via `parent` (which is what new-format Checkpointer
# files contain). `parent` keeps the underlying storage backend
# (Array or CuArray) intact — important for GPU correctness.
function load_interior(data, target_size)
    arr = data isa AbstractArray ? parent(data) : data
    sz = size(arr)
    Hx = (sz[1] - target_size[1]) ÷ 2
    Hy = (sz[2] - target_size[2]) ÷ 2
    Hz = (sz[3] - target_size[3]) ÷ 2
    return arr[Hx+1:Hx+target_size[1], Hy+1:Hy+target_size[2], Hz+1:Hz+target_size[3]]
end

function run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
    if first(closure) isa SmagorinskyLilly
        name = "ocean_large_eddy_simulation_SmagorinskyLilly"
    elseif first(closure) isa DynamicSmagorinsky
        averaging = first(closure).coefficient.averaging
        if averaging isa LagrangianAveraging
            name = "ocean_large_eddy_simulation_DynamicSmagorinsky_lagrangian"
        else
            name = "ocean_large_eddy_simulation_DynamicSmagorinsky_directional"
        end
    else
        firstclosure = first(closure)
        closurename = typeof(firstclosure).name.wrapper
        closurestr = string(closurename)
        name = "ocean_large_eddy_simulation_$closurestr"
    end

    spinup_steps = 10000
      test_steps = 10
              Δt = 2.0

    # Parameters
      Qᵀ = 5e-5     # Temperature flux at surface
      Qᵘ = -2e-5    # Velocity flux at surface
    ∂T∂z = 0.005    # Initial vertical temperature gradient

    # Grid
    N = L = 16
    if grid_type == :regular
        grid = RectilinearGrid(arch, size=(N, N, N), extent=(L, L, L), halo=(2, 2, 2))
    elseif grid_type == :vertically_unstretched
        zF = range(-L, 0, length=N+1)
        grid = RectilinearGrid(arch, size=(N, N, N), x=(0, L), y=(0, L), z=zF, halo=(2, 2, 2))
    end

    # Boundary conditions
    u_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux(), Qᵘ))
    T_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux(), Qᵀ), bottom = BoundaryCondition(Gradient(), ∂T∂z))
    S_bcs = FieldBoundaryConditions(top = BoundaryCondition(Flux(), 5e-8))

    equation_of_state = LinearEquationOfState(thermal_expansion=2e-4, haline_contraction=8e-4)

    # Model instantiation
    model = NonhydrostaticModel(grid; closure,
                                timestepper = :QuasiAdamsBashforth2,
                                coriolis = FPlane(f=1e-4),
                                buoyancy = SeawaterBuoyancy(; equation_of_state),
                                tracers = (:T, :S),
                                hydrostatic_pressure_anomaly = CenterField(grid),
                                boundary_conditions = (u=u_bcs, T=T_bcs, S=S_bcs))

    # The type of the underlying data, not the offset array.
    ArrayType = typeof(parent(model.velocities.u))
    nx, ny, nz = size(model.tracers.T)

    u, v, w = model.velocities
    T, S = model.tracers

    ####
    #### Uncomment the block below to generate regression data.
    ####

    #=
    @warn "Generating new data for the ocean LES regression test."

    # Initialize model: random noise damped at top and bottom
    Ξ(z) = randn() * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
    T₀(x, y, z) = 20 + ∂T∂z * z + ∂T∂z * model.grid.Lz * 1e-2 * Ξ(z)
    u₀(x, y, z) = sqrt(abs(Qᵘ)) * 1e-3 * Ξ(z)
    set!(model, u=u₀, w=u₀, T=T₀, S=35)

    simulation = Simulation(model, Δt=Δt, stop_iteration=spinup_steps-test_steps)
    run!(simulation)

    checkpointer = Checkpointer(model, schedule = IterationInterval(test_steps), prefix = name,
                                dir = joinpath(dirname(@__FILE__), "data"))

    simulation.output_writers[:checkpointer] = checkpointer

    simulation.stop_iteration += 2test_steps
    run!(simulation)
    pop!(simulation.output_writers, :checkpointer)
    =#

    ####
    #### Regression test
    ####

    datadep_path = "regression_truth_data/" * name * "_iteration$spinup_steps.jld2"
    initial_filename = @datadep_str datadep_path

    solution₀, Gⁿ₀, G⁻₀, closure₀, pNHS₀ = get_fields_from_checkpoint(initial_filename)

    Nz = grid.Nz

    cpu_arch = cpu_architecture(architecture(grid))

    u₀ = partition(ArrayType(load_interior(solution₀.u, size(u))), cpu_arch, size(u))
    v₀ = partition(ArrayType(load_interior(solution₀.v, size(v))), cpu_arch, size(v))
    w₀ = partition(ArrayType(load_interior(solution₀.w, size(w))), cpu_arch, size(w))
    T₀ = partition(ArrayType(load_interior(solution₀.T, size(T))), cpu_arch, size(T))
    S₀ = partition(ArrayType(load_interior(solution₀.S, size(S))), cpu_arch, size(S))

    Gⁿu₀ = partition(load_interior(ArrayType(Gⁿ₀.u), size(u)), cpu_arch, size(u))
    Gⁿv₀ = partition(load_interior(ArrayType(Gⁿ₀.v), size(v)), cpu_arch, size(v))
    Gⁿw₀ = partition(load_interior(ArrayType(Gⁿ₀.w), size(w)), cpu_arch, size(w))
    GⁿT₀ = partition(load_interior(ArrayType(Gⁿ₀.T), size(T)), cpu_arch, size(T))
    GⁿS₀ = partition(load_interior(ArrayType(Gⁿ₀.S), size(S)), cpu_arch, size(S))

    G⁻u₀ = partition(load_interior(ArrayType(G⁻₀.u), size(u)), cpu_arch, size(u))
    G⁻v₀ = partition(load_interior(ArrayType(G⁻₀.v), size(v)), cpu_arch, size(v))
    G⁻w₀ = partition(load_interior(ArrayType(G⁻₀.w), size(w)), cpu_arch, size(w))
    G⁻T₀ = partition(load_interior(ArrayType(G⁻₀.T), size(T)), cpu_arch, size(T))
    G⁻S₀ = partition(load_interior(ArrayType(G⁻₀.S), size(S)), cpu_arch, size(S))

    set!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

    interior(model.timestepper.Gⁿ.u) .= Gⁿu₀
    interior(model.timestepper.Gⁿ.v) .= Gⁿv₀
    interior(model.timestepper.Gⁿ.w) .= Gⁿw₀
    interior(model.timestepper.Gⁿ.T) .= GⁿT₀
    interior(model.timestepper.Gⁿ.S) .= GⁿS₀

    interior(model.timestepper.G⁻.u) .= G⁻u₀
    interior(model.timestepper.G⁻.v) .= G⁻v₀
    interior(model.timestepper.G⁻.w) .= G⁻w₀
    interior(model.timestepper.G⁻.T) .= G⁻T₀
    interior(model.timestepper.G⁻.S) .= G⁻S₀

    # Restore closure prognostic state (𝒥ᴸᴹ, 𝒥ᴹᴹ, ..., previous_compute_time)
    # so the closure picks up where the spinup left off rather than bootstrapping
    # fresh. Saved structure: Tuple (one entry per closure) of NamedTuples whose
    # leaves are either (data = OffsetArray,) Field structs or scalar Refs.
    if closure₀ !== nothing
        for (saved, current) in zip(closure₀, model.closure_fields)
            saved isa NamedTuple || continue
            for name in keys(saved)
                hasproperty(current, name) || continue
                target = getproperty(current, name)
                leaf = saved[name]
                if target isa Oceananigans.Fields.Field
                    arr = leaf isa NamedTuple && haskey(leaf, :data) ? leaf.data :
                          leaf isa AbstractArray ? leaf : nothing
                    arr === nothing && continue
                    # ArrayType(...) transfers to the field's backend (CPU/GPU)
                    # before broadcast-assigning to the GPU/CPU interior.
                    interior(target) .= ArrayType(load_interior(arr, size(target)))
                elseif target isa Base.RefValue
                    # e.g. previous_compute_time — restore the scalar so
                    # Δt_lagrangian = clock.time - previous_compute_time
                    # matches the running simulation on the very next step.
                    target[] = leaf
                end
            end
        end
    end

    # Restore non-hydrostatic pressure so the next step's velocity
    # correction starts from the same pressure field as the reference.
    if pNHS₀ !== nothing
        interior(model.pressures.pNHS) .= ArrayType(load_interior(pNHS₀, size(model.pressures.pNHS)))
    end

    model.clock.time = spinup_steps * Δt
    model.clock.iteration = spinup_steps

    update_state!(model)
    model.clock.last_Δt = Δt

    for n in 1:test_steps
        time_step!(model, Δt, euler=false)
    end

    datadep_path = "regression_truth_data/" * name * "_iteration$(spinup_steps+test_steps).jld2"
    final_filename = @datadep_str datadep_path

    solution₁, Gⁿ₁, G⁻₁, _, _ = get_fields_from_checkpoint(final_filename)

    test_fields = @allowscalar (u = Array(interior(model.velocities.u)),
                                v = Array(interior(model.velocities.v)),
                                w = Array(interior(model.velocities.w)[:, :, 1:nz]),
                                T = Array(interior(model.tracers.T)),
                                S = Array(interior(model.tracers.S)))

    u₁ = partition(load_interior(solution₁.u, size(u)), cpu_arch, size(u))
    v₁ = partition(load_interior(solution₁.v, size(v)), cpu_arch, size(v))
    w₁ = partition(load_interior(solution₁.w, size(test_fields.w)), cpu_arch, size(test_fields.w))
    T₁ = partition(load_interior(solution₁.T, size(T)), cpu_arch, size(T))
    S₁ = partition(load_interior(solution₁.S, size(S)), cpu_arch, size(S))

    @show size(test_fields.w), size(w₁)

    correct_fields = (u = u₁,
                      v = v₁,
                      w = w₁,
                      T = T₁,
                      S = S₁)

    summarize_regression_test(test_fields, correct_fields)

    @test all(test_fields.u .≈ correct_fields.u)
    @test all(test_fields.v .≈ correct_fields.v)
    @test all(test_fields.w .≈ correct_fields.w)
    @test all(test_fields.T .≈ correct_fields.T)
    @test all(test_fields.S .≈ correct_fields.S)

    return nothing
end
