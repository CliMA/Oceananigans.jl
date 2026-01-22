using Oceananigans.DistributedComputations: reconstruct_global_field, all_reduce
using Oceananigans.Grids

using JLD2

function Δ_min(grid)
    Δx_min = minimum_xspacing(grid, Center(), Center(), Center())
    Δy_min = minimum_yspacing(grid, Center(), Center(), Center())
    return min(Δx_min, Δy_min)
end

@inline Gaussian(x, y, L) = exp(-(x^2 + y^2) / L^2)

function run_hydrostatic_rotation_regression_test(grid, closure, timestepper; regenerate_data = false)

    g = Oceananigans.defaults.gravitational_acceleration
    R = grid.radius
    Ω = Oceananigans.defaults.planet_rotation_rate

    # Add some shear on the velocity field
    uᵢ(λ, φ, z) = 0.1 * cosd(φ) * sind(λ) + 0.05 * z
    ηᵢ(λ, φ, z) = (R * Ω * 0.1 + 0.1^2 / 2) * sind(φ)^2 / g * sind(λ)
    cᵢ(λ, φ, z) = max(Gaussian(λ, φ - 5, 10), 0.1)
    vᵢ(λ, φ, z) = 0.1

    coriolis = HydrostaticSphericalCoriolis(rotation_rate = Ω)
    free_surface = SplitExplicitFreeSurface(grid; substeps = 20, gravitational_acceleration = g)

    model = HydrostaticFreeSurfaceModel(grid;
                                        momentum_advection = WENOVectorInvariant(order=5),
                                        free_surface,
                                        coriolis,
                                        closure,
                                        tracers = (:c, :b),
                                        timestepper,
                                        tracer_advection = WENO(),
                                        buoyancy = BuoyancyTracer())

    set!(model, u=uᵢ, c=cᵢ, η=ηᵢ)

    # CFL capped Δt
    Δt_local = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz)
    Δt = all_reduce(min, Δt_local, architecture(grid))

    stop_iteration = 20
    simulation = Simulation(model; Δt, stop_iteration, verbose=false)

    coord_str = grid.z isa MutableVerticalDiscretization ? "Mutable" : "Static"
    closure_str = isnothing(closure) ? "Nothing" : "CATKE"
    timestepper_str = timestepper == :QuasiAdamsBashforth2 ? "AB2" : "RK3"

    output_filename = "hydrostatic_rotation_regression_$(coord_str)_$(closure_str)_$(timestepper_str).jld2"

    u, v, w = model.velocities
    c, b = model.tracers
    η = model.free_surface.displacement

    if regenerate_data && !(grid isa DistributedGrid) # never regenerate on Distributed
        @warn "Generating new data for the Hydrostatic regression test."

        directory =  joinpath(dirname(@__FILE__), "data")
        outputs   = (; u, v, w, c, b, η)
        simulation.output_writers[:fields] = JLD2Writer(model, outputs,
                                                        dir = directory,
                                                        schedule = IterationInterval(stop_iteration),
                                                        filename = output_filename,
                                                        with_halos = false,
                                                        overwrite_existing = true)
    end

    run!(simulation)

    # Test results
    test_fields = (
        u = Array(interior(reconstruct_global_field(u))),
        v = Array(interior(reconstruct_global_field(v))),
        w = Array(interior(reconstruct_global_field(w))),
        c = Array(interior(reconstruct_global_field(c))),
        b = Array(interior(reconstruct_global_field(b))),
        η = Array(interior(reconstruct_global_field(η)))
    )

    if !regenerate_data
        datadep_path = "regression_truth_data/" * output_filename
        regression_data_path = @datadep_str datadep_path
        file = jldopen(regression_data_path)

        # Data was saved with 2 halos per direction (see issue #3260)
        truth_fields = (
            u = file["timeseries/u/$stop_iteration"],
            v = file["timeseries/v/$stop_iteration"],
            w = file["timeseries/w/$stop_iteration"],
            η = file["timeseries/η/$stop_iteration"]
        )

        close(file)
    else
        truth_fields = test_fields
    end

    summarize_regression_test(test_fields, truth_fields)

    @test all(test_fields.u .≈ truth_fields.u)
    @test all(test_fields.v .≈ truth_fields.v)
    @test all(test_fields.w .≈ truth_fields.w)
    @test all(test_fields.η .≈ truth_fields.η)
    @test all(test_fields.c .≈ truth_fields.c)
    @test all(test_fields.b .≈ truth_fields.b)

    return nothing
end
