using Statistics
using Oceananigans.Fields: CenterField, ZFaceField, compute_at!, @compute
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: halo_size

include("utils_for_runtests.jl")

archs = test_architectures()

@testset "Averaged fields" begin
    @info "Testing averaged fields..."

    for arch in archs
        @testset "Averaged fields [$(typeof(arch))]" begin
            @info "  Testing AveragedFields [$(typeof(arch))]"
            for FT in float_types

                grid = RegularRectilinearGrid(topology = (Periodic, Periodic, Bounded),
                                              size = (2, 2, 2),
                                              x = (0, 2),
                                              y = (0, 2),
                                              z = (0, 2))

                w = ZFaceField(arch, grid)
                T = CenterField(arch, grid)

                trilinear(x, y, z) = x + y + z

                set!(T, trilinear)
                set!(w, trilinear)

                @compute Txyz = AveragedField(T, dims=(1, 2, 3))

                # Note: halo regions must be *filled* prior to computing an average
                # if the average within halo regions is to be correct.
                fill_halo_regions!(T, arch)
                @compute Txy = AveragedField(T, dims=(1, 2))

                fill_halo_regions!(T, arch)
                @compute Tx = AveragedField(T, dims=1)

                @compute wxyz = AveragedField(w, dims=(1, 2, 3))
                @compute wxy = AveragedField(w, dims=(1, 2))
                @compute wx = AveragedField(w, dims=1)

                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

                @test Txyz[1, 1, 1] ≈ 3

                @test Array(interior(Txy))[1, 1, :] ≈ [2.5, 3.5]
                @test Array(interior(Tx))[1, :, :] ≈ [[2, 3] [3, 4]]

                @test wxyz[1, 1, 1] ≈ 3

                @test Array(interior(wxy))[1, 1, :] ≈ [2, 3, 4]
                @test Array(interior(wx))[1, :, :] ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
                
                # Test whether a race condition gets hit for averages over large fields
                big_grid = RegularRectilinearGrid(topology = (Periodic, Periodic, Bounded),
                                                  size = (256, 256, 128),
                                                  x = (0, 2),
                                                  y = (0, 2),
                                                  z = (0, 2))

                c = CenterField(arch, big_grid)
                c .= 1

                C = AveragedField(c, dims=(1, 2))

                # Test that the mean consistently returns 1 at every z for many evaluations
                results = [all(interior(mean!(C, C.operand)) .== 1) for i = 1:10] # warm up...
                results = [all(interior(mean!(C, C.operand)) .== 1) for i = 1:10] # the real deal
                @test mean(results) == 1.0              
            end
        end

        @testset "Conditional computation of AveragedFields [$(typeof(arch))]" begin
            @info "  Testing conditional computation of AveragedFields [$(typeof(arch))]"
            for FT in float_types
                grid = RegularRectilinearGrid(FT, size=(2, 2, 2), extent=(1, 1, 1))
                c = CenterField(arch, grid)

                for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
                    C = AveragedField(c, dims=dims)

                    @test !isnothing(C.status)

                    # Test conditional computation
                    set!(c, 1)
                    compute_at!(C, FT(1)) # will compute
                    @test all(interior(C) .== 1)
                    @test C.status.time == FT(1)

                    set!(c, 2)
                    compute_at!(C, FT(1)) # will not compute because status == 1
                    @test C.status.time == FT(1)
                    @test all(interior(C) .== 1)

                    compute_at!(C, FT(2))
                    @test C.status.time == FT(2)
                    @test all(interior(C) .== 2)
                end
            end
        end
    end
end
