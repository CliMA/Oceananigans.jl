include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Solvers: HomogeneousNeumannFormulation
using Oceananigans.Grids: XDirection, YDirection, ZDirection

@testset "Fourier tridiagonal Poisson storage history" begin
    for arch in archs, FT in float_types
        for (axis, direction) in ((1, XDirection()), (2, YDirection()), (3, ZDirection()))
            for all_bounded in (false, true)
                topo = ntuple(i -> all_bounded || i == axis ? Bounded : Periodic, 3)
                grid = RectilinearGrid(arch, FT; size=(16, 16, 16), extent=(1, 1, 1), topology=topo)
                formulation = HomogeneousNeumannFormulation(direction)
                solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=formulation)
                source = CenterField(grid)
                solution = CenterField(grid)

                set!(source, (x, y, z) -> sin(2π * x) * cos(2π * y) * (z - 0.5))
                source .= source .- mean(source)
                solve!(solution, solver, source)
                first = Array(interior(solution))

                solve!(solution, solver, source)
                @test Array(interior(solution)) == first

                solver.storage .= one(eltype(solver.storage))
                solve!(solution, solver, source)
                @test Array(interior(solution)) == first

                set!(source, (x, y, z) -> cos(4π * x) * sin(2π * y) * (z - 0.25))
                source .= source .- mean(source)
                solve!(solution, solver, source)

                set!(source, (x, y, z) -> sin(2π * x) * cos(2π * y) * (z - 0.5))
                source .= source .- mean(source)
                solve!(solution, solver, source)
                @test Array(interior(solution)) == first
            end
        end
    end
end
