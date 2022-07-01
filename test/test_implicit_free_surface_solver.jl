include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Architectures: device_event

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    ImplicitFreeSurface,
    FreeSurface,
    FFTImplicitFreeSurfaceSolver,
    PCGImplicitFreeSurfaceSolver,
    MatrixImplicitFreeSurfaceSolver, 
    MGImplicitFreeSurfaceSolver,
    implicit_free_surface_step!,
    implicit_free_surface_linear_operation!

function set_simple_divergent_velocity!(model)
    # Create a divergent velocity
    grid = model.grid

    u, v, w = model.velocities
    η = model.free_surface.η

    u .= 0
    v .= 0
    η .= 0

    imid = Int(floor(grid.Nx / 2)) + 1
    jmid = Int(floor(grid.Ny / 2)) + 1
    CUDA.@allowscalar u[imid, jmid, 1] = 1

    update_state!(model)

    return nothing
end

function run_implicit_free_surface_solver_tests(arch, grid, free_surface)
    Δt = 900

    # Create a model
    model = HydrostaticFreeSurfaceModel(; grid,
                                        momentum_advection = nothing,
                                        free_surface)
    
    events = ((device_event(arch), device_event(arch)), (device_event(arch), device_event(arch)))

    set_simple_divergent_velocity!(model)
    implicit_free_surface_step!(model.free_surface, model, Δt, 1.5, events)

    acronym = free_surface.solver_method == :Multigrid ? "MG" : "PCG"
    
    η = model.free_surface.η
    @info "    " * acronym * " implicit free surface solver test, norm(η_" * lowercase(acronym) * "): $(norm(η)), maximum(abs, η_" * lowercase(acronym) * "): $(maximum(abs, η))"

    # Extract right hand side "truth"
    right_hand_side = model.free_surface.implicit_step_solver.right_hand_side

    # Compute left hand side "solution"
    g = g_Earth
    η = model.free_surface.η
    ∫ᶻ_Axᶠᶜᶜ = model.free_surface.implicit_step_solver.vertically_integrated_lateral_areas.xᶠᶜᶜ
    ∫ᶻ_Ayᶜᶠᶜ = model.free_surface.implicit_step_solver.vertically_integrated_lateral_areas.yᶜᶠᶜ

    left_hand_side = Field{Center, Center, Nothing}(grid)
    implicit_free_surface_linear_operation!(left_hand_side, η, ∫ᶻ_Axᶠᶜᶜ, ∫ᶻ_Ayᶜᶠᶜ, g, Δt)

    # Compare
    extrema_tolerance = 1e-9
    std_tolerance = 1e-9

    CUDA.@allowscalar begin
        @test maximum(abs, interior(left_hand_side) .- interior(right_hand_side)) < extrema_tolerance
        @test std(interior(left_hand_side) .- interior(right_hand_side)) < std_tolerance
    end

    return nothing
end

@testset "Implicit free surface solver tests" begin
    for arch in archs
        A = typeof(arch)

        rectilinear_grid = RectilinearGrid(arch, size = (128, 1, 5),
                                           x = (0, 1000kilometers), y = (0, 1), z = (-400, 0),
                                           topology = (Bounded, Periodic, Bounded))

        lat_lon_grid = LatitudeLongitudeGrid(arch, size = (90, 90, 5),
                                             longitude = (-30, 30), latitude = (15, 75), z = (-4000, 0))

        for grid in (rectilinear_grid, lat_lon_grid)
            G = string(nameof(typeof(grid)))

            @info "Testing PreconditionedConjugateGradient implicit free surface solver [$A, $G]..."
            free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, abstol=1e-15, reltol=0)
            run_implicit_free_surface_solver_tests(arch, grid, free_surface)

            @info "Testing Multigrid implicit free surface solver [$A, $G]..."
            free_surface = ImplicitFreeSurface(solver_method=:Multigrid, abstol=1e-15, reltol=0)
            run_implicit_free_surface_solver_tests(arch, grid, free_surface)   
        end

        @info "Testing implicit free surface solvers compared to FFT [$A]..."

        mat_free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver,
                                               tolerance=1e-15, maximum_iterations=128^3)

        pcg_free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient,
                                               abstol=1e-15, reltol=0, maxiter=128^3)

        mg_free_surface = ImplicitFreeSurface(solver_method=:Multigrid,
                                              abstol=1e-15, reltol=0, maxiter=128^3)

        fft_free_surface = ImplicitFreeSurface(solver_method=:FastFourierTransform)

        mat_model = HydrostaticFreeSurfaceModel(grid = rectilinear_grid,
                                                momentum_advection = nothing,
                                                free_surface = mat_free_surface)

        pcg_model = HydrostaticFreeSurfaceModel(grid = rectilinear_grid,
                                                momentum_advection = nothing,
                                                free_surface = pcg_free_surface)

        mg_model = HydrostaticFreeSurfaceModel(grid = rectilinear_grid,
                                               momentum_advection = nothing,
                                               free_surface = mg_free_surface)

        fft_model = HydrostaticFreeSurfaceModel(grid = rectilinear_grid,
                                                momentum_advection = nothing,
                                                free_surface = fft_free_surface)

        @test fft_model.free_surface.implicit_step_solver isa FFTImplicitFreeSurfaceSolver
        @test pcg_model.free_surface.implicit_step_solver isa PCGImplicitFreeSurfaceSolver
        @test mat_model.free_surface.implicit_step_solver isa MatrixImplicitFreeSurfaceSolver
        @test  mg_model.free_surface.implicit_step_solver isa MGImplicitFreeSurfaceSolver
        
        events = ((device_event(arch), device_event(arch)), (device_event(arch), device_event(arch)))

        Δt = 900

        for m in (mat_model, pcg_model, fft_model, mg_model)
            set_simple_divergent_velocity!(m)
            implicit_free_surface_step!(m.free_surface, m, Δt, 1.5, events)
        end

        mat_η = mat_model.free_surface.η
        pcg_η = pcg_model.free_surface.η
        fft_η = fft_model.free_surface.η
        mg_η  =  mg_model.free_surface.η

        mat_η_cpu = Array(interior(mat_η))
        pcg_η_cpu = Array(interior(pcg_η))
        fft_η_cpu = Array(interior(fft_η))
        mg_η_cpu  = Array(interior(mg_η))

        Δη_mat = mat_η_cpu .- fft_η_cpu
        Δη_pcg = pcg_η_cpu .- fft_η_cpu
        Δη_mg  = mg_η_cpu  .- fft_η_cpu

        @info "FFT/PCG/MAT/MG implicit free surface solver comparison:"
        @info "    maximum(abs, η_mat - η_fft): $(maximum(abs, Δη_mat))"
        @info "    maximum(abs, η_pcg - η_fft): $(maximum(abs, Δη_pcg))"
        @info "    maximum(abs, η_mg - η_fft) : $(maximum(abs, Δη_mg))"
        @info "    maximum(abs, η_mat): $(maximum(abs, mat_η_cpu))"
        @info "    maximum(abs, η_pcg): $(maximum(abs, pcg_η_cpu))"
        @info "    maximum(abs, η_mg) : $(maximum(abs, mg_η_cpu))"
        @info "    maximum(abs, η_fft): $(maximum(abs, fft_η_cpu))"

        @test all(mat_η_cpu .≈ fft_η_cpu)
        
        if arch isa CPU
            @test all(pcg_η_cpu .≈ fft_η_cpu)
            @test all(mg_η_cpu .≈ fft_η_cpu)
        else
            # It seems that the PCG algorithm is not always stable on sverdrup's GPU, often leading to failure.
            # This behavior is not observed on tartarus, where this test _would_ pass.
            # Suffice to say that the FFT solver appears to be accurate (as of this writing), and tests pass
            # on the CPU.
            @info "  Skipping comparison between pcg and fft implicit free surface solver"
            @test_skip all(pcg_η_cpu .≈ fft_η_cpu)
            @info "  Skipping comparison between mg and fft implicit free surface solver"
            @test_skip all(mg_η_cpu .≈ fft_η_cpu)
        end

        @test all(isapprox.(Δη_mat, 0, atol=sqrt(eps(eltype(rectilinear_grid)))))
        @test all(isapprox.(Δη_pcg, 0, atol=sqrt(eps(eltype(rectilinear_grid)))))
        @test all(isapprox.(Δη_mg,  0, atol=sqrt(eps(eltype(rectilinear_grid)))))
    end
end
