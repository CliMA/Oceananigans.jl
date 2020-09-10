using Statistics

using Oceananigans.Fields: CellField, ZFaceField
using Oceananigans.Grids: halo_size

@testset "Averaged fields" begin
    @info "Testing averaged fields..."

    for arch in archs
        @testset "Averaged fields [$(typeof(arch))]" begin
            @info "  Testing AveragedFields [$(typeof(arch))]"
            for FT in float_types

                grid = RegularCartesianGrid(topology = (Periodic, Periodic, Bounded),
                                                size = (2, 2, 2),
                                                   x = (0, 2), y = (0, 2), z = (0, 2))

                w = ZFaceField(arch, grid)
                T = CellField(arch, grid)

                trilinear(x, y, z) = x + y + z

                set!(T, trilinear)
                set!(w, trilinear)

                @compute T̃ = mean(T, dims=(1, 2, 3))
                @compute T̅ = mean(T, dims=(1, 2))
                @compute T̂ = mean(T, dims=1)

                @compute w̃ = mean(w, dims=(1, 2, 3))
                @compute w̅ = mean(w, dims=(1, 2))
                @compute ŵ = mean(w, dims=1)

                Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

                @test T̃[1, 1, 1] ≈ 3
                @test Array(T̅[1, 1, 1:Nz]) ≈ [2.5, 3.5]
                @test Array(T̂[1, 1:Ny, 1:Nz]) ≈ [[2, 3] [3, 4]]

                @test w̃[1, 1, 1] ≈ 4.5
                @test Array(w̅[1, 1, 1:Nz+1]) ≈ [2, 3, 4]
                @test Array(ŵ[1, 1:Ny, 1:Nz+1]) ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
            end
        end
    end
end
