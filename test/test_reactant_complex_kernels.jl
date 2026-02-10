include("reactant_test_utils.jl")

using Oceananigans.Architectures: ReactantState
using Oceananigans.Models.NonhydrostaticModels: compute_source_term!
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.Fields: interior

#####
##### Tests for FFT solver kernels that previously hit the ComplexF64 MLIR bug (B.6.7).
##### The workaround routes real↔complex conversions through broadcasts instead of KA kernels.
##### These tests verify that the kernels compile and raise with raise=true.
#####

@testset "Reactant ComplexF64 kernel workarounds" begin
    @info "Performing Reactant ComplexF64 kernel workaround tests..."
    reactant_arch = ReactantState()

    ####
    #### FFTBasedPoissonSolver (triply periodic / 2D periodic grids)
    ####

    @testset "FFTBasedPoissonSolver (Periodic, Periodic, Flat)" begin
        @info "  Setting up FFTBased 2D (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4), extent=(1, 1),
                               halo=(3, 3), topology=(Periodic, Periodic, Flat))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        solver = model.pressure_solver

        @test solver isa FFTBasedPoissonSolver

        @testset "compute_source_term! (Issue 1: Float64 → ComplexF64)" begin
            @info "    Testing compute_source_term! with raise=true..."
            function test_fft_source_2d!(solver, velocities)
                compute_source_term!(solver, nothing, velocities, 1.0)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_fft_source_2d!(solver, model.velocities)
            compiled!(solver, model.velocities)
            @test true
        end

        @testset "solve! (Issue 2: real from ComplexF64)" begin
            @info "    Testing solve! with raise=true..."
            pressure = model.pressures.pNHS
            function test_fft_solve_2d!(pressure, solver)
                solve!(pressure, solver)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_fft_solve_2d!(pressure, solver)
            compiled!(pressure, solver)
            @test true
        end
    end

    @testset "FFTBasedPoissonSolver (Periodic, Periodic, Periodic)" begin
        @info "  Setting up FFTBased 3D (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; timestepper=:RungeKutta3)
        solver = model.pressure_solver

        @testset "compute_source_term! (Issue 1: Float64 → ComplexF64)" begin
            @info "    Testing compute_source_term! with raise=true..."
            function test_fft_source_3d!(solver, velocities)
                compute_source_term!(solver, nothing, velocities, 1.0)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_fft_source_3d!(solver, model.velocities)
            compiled!(solver, model.velocities)
            @test true
        end

        @testset "solve! (Issue 2: real from ComplexF64)" begin
            @info "    Testing solve! with raise=true..."
            pressure = model.pressures.pNHS
            function test_fft_solve_3d!(pressure, solver)
                solve!(pressure, solver)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_fft_solve_3d!(pressure, solver)
            compiled!(pressure, solver)
            @test true
        end
    end

    #####
    ##### FourierTridiagonalPoissonSolver (grids with one stretched direction)
    ##### Uses (Periodic, Periodic, Bounded) with stretched z to trigger FourierTridiagonal.
    #####

    @testset "FourierTridiagonalPoissonSolver (Periodic, Periodic, Bounded)" begin
        @info "  Setting up FourierTridiagonal (Periodic, Periodic, Bounded) with stretched z..."
        # Stretched z-faces produce a non-regular z → XYRegularRG → FourierTridiagonalPoissonSolver
        z_faces = [0, 0.2, 0.5, 0.8, 1.0]
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=z_faces,
                               topology=(Periodic, Periodic, Bounded))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        solver = model.pressure_solver

        @test solver isa FourierTridiagonalPoissonSolver

        @testset "compute_source_term! (Issue 1: Float64 → ComplexF64)" begin
            @info "    Testing compute_source_term! with raise=true..."
            function test_ftps_source!(solver, velocities)
                compute_source_term!(solver, nothing, velocities, 1.0)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_ftps_source!(solver, model.velocities)
            compiled!(solver, model.velocities)
            @test true
        end

        @testset "solve! (Issue 2: real from ComplexF64)" begin
            @info "    Testing solve! with raise=true..."
            pressure = model.pressures.pNHS
            function test_ftps_solve!(pressure, solver)
                solve!(pressure, solver)
                return nothing
            end
            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_ftps_solve!(pressure, solver)
            compiled!(pressure, solver)
            @test true
        end
    end

    # Note: (Periodic, Bounded, Flat) with stretched y is skipped because
    # FourierTridiagonalPoissonSolver construction fails on ReactantState due to
    # eager kernel launches during compute_main_diagonal! (B.6.6).
end
