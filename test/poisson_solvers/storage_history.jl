include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Solvers: HomogeneousNeumannFormulation
using Oceananigans.Grids: XDirection, YDirection, ZDirection

@testset "Fourier tridiagonal Poisson storage history" begin
    for arch in archs, FT in float_types
        for (axis, direction) in ((1, XDirection()), (2, YDirection()), (3, ZDirection()))
            topo = ntuple(i -> i == axis ? Bounded : Periodic, 3)
            grid = RectilinearGrid(arch, FT; size=(16, 16, 16), extent=(1, 1, 1), topology=topo)
            formulation = HomogeneousNeumannFormulation(direction)
            solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=formulation)
            first_source = CenterField(grid)
            second_source = CenterField(grid)
            first_solution = CenterField(grid)
            solution = CenterField(grid)

            set!(first_source, (x, y, z) -> sin(2π * x) * cos(2π * y) * (z - 0.5))
            first_source .= first_source .- mean(first_source)
            set!(second_source, (x, y, z) -> cos(4π * x) * sin(2π * y) * (z - 0.25))
            second_source .= second_source .- mean(second_source)

            solve!(first_solution, solver, first_source)
            solve!(solution, solver, first_source)
            @test solution == first_solution

            solve!(solution, solver, second_source)
            solve!(solution, solver, first_source)
            @test solution == first_solution
        end
    end
end
