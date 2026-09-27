include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))
include(joinpath(@__DIR__, "..", "setup", "split_ffsl_test_utils.jl"))

using Oceananigans.BoundaryConditions: fill_halo_regions!

# The stretching stored at the beginning of the long step must equal its halo-filled self over the cells that the
# swept regions reach (Cmax + 1 cells beyond the interior)
function stored_stretching_halo_error(model)
    σ⁰ = flux_form_semi_lagrangian_workspace(model.advection).σ⁰
    filled = Field{Center, Center, Nothing}(model.grid)
    parent(filled) .= parent(σ⁰)
    fill_halo_regions!(filled)
    Nx, Ny, _ = size(model.grid)
    reach = 4
    TX, TY, _ = topology(model.grid)
    ii = TX == Periodic ? (1-reach:Nx+reach) : (1:Nx)
    jj = TY == Bounded ? (1:Ny) : (1:Ny+reach)
    σ⁰ᶜᵖᵘ = on_architecture(CPU(), σ⁰)
    filledᶜᵖᵘ = on_architecture(CPU(), filled)
    return maximum(abs(filledᶜᵖᵘ[i, j, 1] - σ⁰ᶜᵖᵘ[i, j, 1]) for i in ii, j in jj)
end

slow_tracer_data(model) = Tuple(Array(interior(model.tracers[name])) for name in (:c, :constant, :smooth))

function test_split_ffsl_conservation(model, ratio, Δt, cycles; minimum_courant_number, fold_rows = nothing)
    active = active_cells(model.grid)
    inventory₀ = tracer_inventory(model.tracers.c)
    η₀ = Array(interior(model.free_surface.displacement))
    swept_courant = 0.0
    fold_courant = 0.0
    outflow_courant = 0.0

    for cycle in 1:cycles
        slow_data = slow_tracer_data(model)
        for step in 1:ratio
            time_step!(model, Δt)
            # The slow tracers are frozen between long steps
            step < ratio && @test slow_tracer_data(model) == slow_data
        end

        swept_courant = max(swept_courant, maximum(swept_courant_numbers(model)))
        outflow_courant = max(outflow_courant, horizontal_outflow_courant_number(model, ratio * Δt))
        isnothing(fold_rows) || (fold_courant = max(fold_courant, maximum(swept_courant_numbers(model; rows = fold_rows))))
        @test maximum_uniform_deviation(model.tracers.constant, 1, active) < 1e-12
        @test maximum_uniform_deviation(model.tracers.fast, 1, active) < 1e-12
        @test abs(tracer_inventory(model.tracers.c) - inventory₀) / inventory₀ < 1e-12
    end

    @test stored_stretching_halo_error(model) < 1e-14
    @test maximum(abs, Array(interior(model.free_surface.displacement)) .- η₀) > 1e-3
    @test all(isfinite, Array(interior(model.tracers.smooth))[active])
    @test swept_courant > minimum_courant_number

    # FFSL is stable while no cell loses more than its volume horizontally during a long step
    @test outflow_courant < 1

    return fold_courant
end

@testset "Split tracer time stepping with FluxFormSemiLagrangian slow tracers" begin
    for arch in archs
        @testset "Halo validation [$(typeof(arch))]" begin
            @info "  Testing the halo required by FluxFormSemiLagrangian slow tracers [$(typeof(arch))]..."
            @test_throws ArgumentError baroclinic_basin_model(rectilinear_basin(arch; halo = (4, 4, 4)); ratio = 4)
        end

        @testset "Mixed FFSL and WENO slow groups [$(typeof(arch))]" begin
            @info "  Testing mixed FFSL and WENO slow tracers with a fast FFSL tracer [$(typeof(arch))]..."
            grid = rectilinear_basin(arch)
            weno = WENO(order=5)
            ffsl = FluxFormSemiLagrangian()
            splitting = TracerTimeStepSplitting(tracers = (:c, :constant, :weno), ratio = 4)
            model = HydrostaticFreeSurfaceModel(grid; free_surface = SplitExplicitFreeSurface(grid; substeps = 8),
                                                tracer_advection = (b = weno, fast = ffsl, c = ffsl, constant = ffsl, weno = weno),
                                                timestepper = :SplitRungeKutta3, buoyancy = BuoyancyTracer(),
                                                tracers = (:b, :fast, :c, :constant, :weno),
                                                tracer_time_step_splitting = splitting)
            x₀ = 16kilometers
            Random.seed!(1234)
            set!(model, b = (x, y, z) -> x < x₀ ? 0.1 : 0.02, fast = 1, c = (x, y, z) -> rand(), constant = 1, weno = 1)
            active = active_cells(grid)
            inventory₀ = tracer_inventory(model.tracers.c)

            for step in 1:8
                time_step!(model, 3minutes)
            end

            for name in (:fast, :constant, :weno)
                @test maximum_uniform_deviation(model.tracers[name], 1, active) < 1e-12
            end
            @test abs(tracer_inventory(model.tracers.c) - inventory₀) / inventory₀ < 1e-12
        end

        @testset "Rectilinear basin, z-star, long-step Courant number > 2 [$(typeof(arch))]" begin
            @info "  Testing FFSL slow tracers at a long-step Courant number > 2 in a rectilinear basin [$(typeof(arch))]..."
            ratio = 32
            model = nondivergent_gyre_model(rectilinear_basin(arch); ratio)
            test_split_ffsl_conservation(model, ratio, 3minutes, 3; minimum_courant_number = 2)
        end

        for fold_topology in (RightCenterFolded, RightFaceFolded)
            @testset "$fold_topology TripolarGrid, z-star, long-step Courant number > 2 [$(typeof(arch))]" begin
                @info "  Testing FFSL slow tracers across the fold of a $fold_topology TripolarGrid [$(typeof(arch))]..."
                ratio = 32
                grid = tripolar_basin(arch; fold_topology)
                model = tripolar_rotation_model(grid; ratio)
                Ny = size(grid, 2)
                fold_courant = test_split_ffsl_conservation(model, ratio, 3hours, 2; minimum_courant_number = 2, fold_rows = Ny-1:Ny+1)
                @test fold_courant > 1
            end
        end
    end
end
