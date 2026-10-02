include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Solvers: HomogeneousNeumannFormulation
using Oceananigans.Grids: XDirection, YDirection, ZDirection

@testset "Fourier tridiagonal Poisson storage history" begin
    for arch in archs, FT in float_types
        for (axis, direction) in ((1, XDirection()), (2, YDirection()), (3, ZDirection()))
            topo = ntuple(i -> i == axis ? Bounded : Periodic, 3)
            grid = RectilinearGrid(arch, FT; size=(16, 16, 16), extent=(1, 1, 1), topology=topo)
            formulation = HomogeneousNeumannFormulation(direction)
            expected_solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=formulation)
            solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=formulation)
            source = CenterField(grid)
            expected = CenterField(grid)
            solution = CenterField(grid)

            set!(source, (x, y, z) -> sin(2π * x) * cos(2π * y) * (z - 0.5))
            source .= source .- mean(source)
            solve!(expected, expected_solver, source)

            set!(source, (x, y, z) -> cos(4π * x) * sin(2π * y) * (z - 0.25))
            source .= source .- mean(source)
            solve!(solution, solver, source)

            set!(source, (x, y, z) -> sin(2π * x) * cos(2π * y) * (z - 0.5))
            source .= source .- mean(source)
            solve!(solution, solver, source)
            @test solution == expected
        end
    end
end
