using Statistics

using Oceananigans.Fields: CenterField, ZFaceField, compute_at!
using Oceananigans.Grids: halo_size

@testset "Averaged fields" begin
    @info "Testing averaged fields..."

    for arch in archs
        @testset "Averaged fields [$(typeof(arch))]" begin
            @info "  Testing AveragedFields [$(typeof(arch))]"
            for FT in float_types

                grid = RegularRectilinearGrid(topology = (Periodic, Periodic, Bounded),
                                                size = (2, 2, 2),
                                                   x = (0, 2), y = (0, 2), z = (0, 2))

                w = ZFaceField(arch, grid)
                T = CenterField(arch, grid)

                trilinear(x, y, z) = x + y + z

                set!(T, trilinear)
                set!(w, trilinear)

                @compute T̃ = AveragedField(T, dims=(1, 2, 3))

                # Note: halo regions must be *filled* prior to computing an average
                # if the average within halo regions is to be correct.
                fill_halo_regions!(T)
                @compute T̅ = AveragedField(T, dims=(1, 2))

                fill_halo_regions!(T)
                @compute T̂ = AveragedField(T, dims=1)

                @compute w̃ = AveragedField(w, dims=(1, 2, 3))
                @compute w̅ = AveragedField(w, dims=(1, 2))
                @compute ŵ = AveragedField(w, dims=1)

                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

                @test T̃[1, 1, 1] ≈ 3
                @test Array(parent(T̅)[1, 1, :]) ≈ [2.5, 2.5, 3.5, 3.5]
                @test Array(parent(T̂)[1, :, :]) ≈ [[3, 2, 3, 2] [3, 2, 3, 2] [4, 3, 4, 3] [4, 3, 4, 3]]

                @test w̃[1, 1, 1] ≈ 3
                @test Array(w̅[1, 1, 1:Nz+1]) ≈ [2, 3, 4]
                @test Array(ŵ[1, 1:Ny, 1:Nz+1]) ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
            end
        end

        @testset "Conditional computation of AveragedFields [$(typeof(arch))]" begin
            @info "  Testing conditional computation of AveragedFields [$(typeof(arch))]"
            for FT in float_types
                grid = RegularRectilinearGrid(size=(2, 2, 2), extent=(1, 1, 1))
                c = CenterField(FT, arch, grid)

                for dims in (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))
                    C = AveragedField(c, dims=dims)

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
