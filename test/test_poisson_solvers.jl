include("dependencies_for_runtests.jl")
include("dependencies_for_poisson_solvers.jl")

#####
##### Run pressure solver tests 1
#####

PB = (Periodic, Bounded)
topos = collect(Iterators.product(PB, PB, PB))[:]

two_dimensional_topologies = [(Flat,     Bounded,  Bounded),
                              (Bounded,  Flat,     Bounded),
                              (Bounded,  Bounded,  Flat),
                              (Flat,     Periodic, Bounded),
                              (Periodic, Flat,     Bounded),
                              (Periodic, Bounded,  Flat)]

@testset "Poisson solvers 1" begin
    @info "Testing Poisson solvers..."

    for arch in archs
        @testset "Poisson solver instantiation [$(typeof(arch))]" begin
            @info "  Testing Poisson solver instantiation [$(typeof(arch))]..."
            for FT in float_types

                grids_3d = [RectilinearGrid(arch, FT, size=(2, 2, 2), extent=(1, 1, 1)),
                            RectilinearGrid(arch, FT, size=(1, 2, 2), extent=(1, 1, 1)),
                            RectilinearGrid(arch, FT, size=(2, 1, 2), extent=(1, 1, 1)),
                            RectilinearGrid(arch, FT, size=(2, 2, 1), extent=(1, 1, 1))]

                grids_2d = [RectilinearGrid(arch, FT, size=(2, 2), extent=(1, 1), topology=topo)
                            for topo in two_dimensional_topologies]


                grids = []
                push!(grids, grids_3d..., grids_2d...)

                for grid in grids
                    @test poisson_solver_instantiates(grid, FFTW.ESTIMATE)
                    @test poisson_solver_instantiates(grid, FFTW.MEASURE)
                end
            end
        end

        @testset "Divergence-free solution [$(typeof(arch))]" begin
            @info "  Testing divergence-free solution [$(typeof(arch))]..."

            for topo in topos
                for N in [7, 16]

                    grids_3d = [RectilinearGrid(arch, topology=topo, size=(N, N, N), extent=(1, 1, 1)),
                                RectilinearGrid(arch, topology=topo, size=(1, N, N), extent=(1, 1, 1)),
                                RectilinearGrid(arch, topology=topo, size=(N, 1, N), extent=(1, 1, 1)),
                                RectilinearGrid(arch, topology=topo, size=(N, N, 1), extent=(1, 1, 1))]

                    grids_2d = [RectilinearGrid(arch, size=(N, N), extent=(1, 1), topology=topo)
                                for topo in two_dimensional_topologies]

                    grids = []
                    push!(grids, grids_3d..., grids_2d...)

                    for grid in grids
                        N == 7 && @info "    Testing $(topology(grid)) topology on square grids [$(typeof(arch))]..."
                        @test divergence_free_poisson_solution(grid)
                    end
                end
            end

            Ns = [11, 16]
            for topo in topos
                @info "    Testing $topo topology on rectangular grids with even and prime sizes [$(typeof(arch))]..."
                for Nx in Ns, Ny in Ns, Nz in Ns
                    grid = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
                    @test divergence_free_poisson_solution(grid)
                end
            end

            # Do a couple at Float32 (since its too expensive to repeat all tests...)
            Float32_grids = [RectilinearGrid(arch, Float32, topology=(Periodic, Bounded, Bounded), size=(16, 16, 16), extent=(1, 1, 1)),
                             RectilinearGrid(arch, Float32, topology=(Bounded, Bounded, Periodic), size=(7, 11, 13), extent=(1, 1, 1))]

            for grid in Float32_grids
                @test divergence_free_poisson_solution(grid)
            end
        end

        @testset "Convergence to analytic solution [$(typeof(arch))]" begin
            @info "  Testing convergence to analytic solution [$(typeof(arch))]..."
            for topo in topos
                @test poisson_solver_convergence(arch, topo, 2^6, 2^7)
                @test poisson_solver_convergence(arch, topo, 67, 131, mode=2)
            end
        end
    end
end
