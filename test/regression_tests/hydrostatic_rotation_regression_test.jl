# include("../dependencies_for_runtests.jl")
# include("../data_dependencies.jl")

using Oceananigans.DistributedComputations: @root
using Oceananigans.Grids: topology, XRegularLLG, YRegularLLG, ZRegularLLG
using Oceananigans.DistributedComputations: synchronized, Distributed
using Oceananigans.DistributedComputations: DistributedGrid, reconstruct_global_field, all_reduce, @root, reconstruct_global_grid
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

    Ψ(y) = - tanh(y)
    U(y) = sech(y)^2

    # A sinusoidal tracer
    C(y, L) = sin(2π * y / L)

    # Slightly off-center vortical perturbations
    ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

    # Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃ 
    ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2)
    ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x)

    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    dr(x) = deg2rad(x)

    global_grid = reconstruct_global_grid(grid)

    # Total initial conditions with some shear
    uᵢ(x, y, z) = (U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)) * (grid.Lz + z) / grid.Lz
    vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k) * (grid.Lz + z) / grid.Lz
    cᵢ(x, y, z) = C(dr(y)*8, global_grid.Ly)

    # Meridional stratification
    bᵢ(x, y, z) = y > 0 ? 0.01 : 0.06

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

    set!(model, u=uᵢ, v=vᵢ, c=cᵢ, b=bᵢ)

    # CFL capped Δt
    Δt_local = 0.1 * Δ_min(grid) / sqrt(g * grid.Lz)
    Δt = all_reduce(min, Δt_local, architecture(grid))

    stop_iteration = 40
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
                                                        array_type = Array{Float64},
                                                        overwrite_existing = true)
    end

    run!(simulation)

    # We remove the w-surface level for a ZStar model
    # since the surface is already a machine precision (~1e-19) solution
    # so it will not pass the test even if the solutions is correct
    idxw = if model.vertical_coordinate isa ZStarCoordinate
        (:, :, 1:size(grid, 3))
    else
        (:, :, :)
    end

    # Test results
    test_fields = (
        u = Array(interior(reconstruct_global_field(u))),
        v = Array(interior(reconstruct_global_field(v))),
        w = Array(interior(reconstruct_global_field(w), idxw...)),
        c = Array(interior(reconstruct_global_field(c))),
        b = Array(interior(reconstruct_global_field(b))),
        η = Array(interior(reconstruct_global_field(η)))
    )

    if !regenerate_data
        datadep_path = "regression_truth_data/" * output_filename
        regression_data_path = @datadep_str datadep_path
        file = jldopen(regression_data_path)

        truth_fields = (
            u = file["timeseries/u/$stop_iteration"],
            v = file["timeseries/v/$stop_iteration"],
            w = file["timeseries/w/$stop_iteration"][idxw...],
            c = file["timeseries/c/$stop_iteration"],
            b = file["timeseries/b/$stop_iteration"],
            η = file["timeseries/η/$stop_iteration"]
        )

        close(file)
    else
        truth_fields = test_fields
    end

    @root summarize_regression_test(test_fields, truth_fields)

    @test all(test_fields.u .≈ truth_fields.u)
    @test all(test_fields.v .≈ truth_fields.v)
    @test all(test_fields.w .≈ truth_fields.w)
    @test all(test_fields.η .≈ truth_fields.η)
    @test all(test_fields.c .≈ truth_fields.c)
    @test all(test_fields.b .≈ truth_fields.b)

    return nothing
end
