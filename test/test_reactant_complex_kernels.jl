include("reactant_test_utils.jl")

using Oceananigans.Architectures: ReactantState
using Oceananigans.Models.NonhydrostaticModels: compute_source_term!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!
using Oceananigans.Fields: interior

#####
##### Tests for FFTBasedPoissonSolver kernels that previously hit the ComplexF64 MLIR bug (B.6.7).
##### The workaround routes real↔complex conversions through broadcasts instead of KA kernels.
##### These tests verify that the kernels compile and raise with raise=true.
#####

@testset "Reactant ComplexF64 kernel workarounds" begin
    @info "Performing Reactant ComplexF64 kernel workaround tests..."
    reactant_arch = ReactantState()

    @testset "FFTBasedPoissonSolver (Periodic, Periodic, Flat)" begin
        @info "  Setting up 2D (Periodic, Periodic, Flat)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4), extent=(1, 1),
                               halo=(3, 3), topology=(Periodic, Periodic, Flat))
        model = NonhydrostaticModel(grid; timestepper=:QuasiAdamsBashforth2)
        solver = model.pressure_solver

        @test solver isa FFTBasedPoissonSolver
        @test solver.scratch !== nothing

        # Issue 1: compute_source_term! writes Float64 into scratch, then broadcasts into ComplexF64 storage.
        @testset "compute_source_term! (Issue 1: Float64 → ComplexF64)" begin
            @info "    Testing compute_source_term! compilation with raise=true..."

            function test_compute_source_term!(solver, velocities)
                compute_source_term!(solver, nothing, velocities, 1.0)
                return nothing
            end

            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_compute_source_term!(solver, model.velocities)
            compiled!(solver, model.velocities)
            @test true
        end

        # Issue 2: solve! extracts real component via broadcast instead of KA kernel.
        @testset "solve! with interior .= real. (Issue 2: real from ComplexF64)" begin
            @info "    Testing solve! compilation with raise=true..."

            pressure = model.pressures.pNHS

            function test_solve!(pressure, solver)
                solve!(pressure, solver)
                return nothing
            end

            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_solve!(pressure, solver)
            compiled!(pressure, solver)
            @test true
        end
    end

    @testset "FFTBasedPoissonSolver (Periodic, Periodic, Periodic)" begin
        @info "  Setting up 3D (Periodic, Periodic, Periodic)..."
        grid = RectilinearGrid(reactant_arch; size=(4, 4, 4), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Periodic))
        model = NonhydrostaticModel(grid; timestepper=:RungeKutta3)
        solver = model.pressure_solver

        @testset "compute_source_term! (Issue 1: Float64 → ComplexF64)" begin
            @info "    Testing compute_source_term! compilation with raise=true..."

            function test_compute_source_term_3d!(solver, velocities)
                compute_source_term!(solver, nothing, velocities, 1.0)
                return nothing
            end

            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_compute_source_term_3d!(solver, model.velocities)
            compiled!(solver, model.velocities)
            @test true
        end

        @testset "solve! with interior .= real. (Issue 2: real from ComplexF64)" begin
            @info "    Testing solve! compilation with raise=true..."

            pressure = model.pressures.pNHS

            function test_solve_3d!(pressure, solver)
                solve!(pressure, solver)
                return nothing
            end

            compiled! = Reactant.@compile raise_first=true raise=true sync=true test_solve_3d!(pressure, solver)
            compiled!(pressure, solver)
            @test true
        end
    end
end
