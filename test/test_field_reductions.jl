include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.Architectures: arch_array
using Oceananigans.Fields: ReducedField, CenterField, ZFaceField, compute_at!, @compute
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: halo_size

@testset "Fields computed by Reduction" begin
    @info "Testing Fields computed by reductions..."

    for arch in archs
        arch_str = string(typeof(arch))

        @testset "Averaged fields [$arch_str]" begin
            @info "  Testing averaged Fields [$arch_str]"

            grid = RectilinearGrid(arch,
                                   topology = (Periodic, Periodic, Bounded),
                                   size = (2, 2, 2),
                                   x = (0, 2),
                                   y = (0, 2),
                                   z = (0, 2))

            Nx, Ny, Nz = size(grid)

            w = ZFaceField(grid)
            T = CenterField(grid)

            trilinear(x, y, z) = x + y + z

            set!(T, trilinear)
            set!(w, trilinear)

            #@compute Txyz = AveragedField(T, dims=(1, 2, 3))
            @compute Txyz = Field(Average(T, dims=(1, 2, 3)))

            # Note: halo regions must be *filled* prior to computing an average
            # if the average within halo regions is to be correct.
            fill_halo_regions!(T, arch)
            #@compute Txy = AveragedField(T, dims=(1, 2))
            @compute Txy = Field(Average(T, dims=(1, 2)))

            fill_halo_regions!(T, arch)
            @compute Tx = AveragedField(T, dims=1)

            @compute wxyz = Field(Average(w, dims=(1, 2, 3)))
            @compute wxy = Field(Average(w, dims=(1, 2)))
            @compute wx = Field(Average(w, dims=1))

            for FT in float_types
                
                @test Txyz[1, 1, 1] ≈ 3

                @test Array(interior(Txy))[1, 1, :] ≈ [2.5, 3.5]
                @test Array(interior(Tx))[1, :, :] ≈ [[2, 3] [3, 4]]

                @test wxyz[1, 1, 1] ≈ 3

                @test Array(interior(wxy))[1, 1, :] ≈ [2, 3, 4]
                @test Array(interior(wx))[1, :, :] ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
                
                # Test whether a race condition gets hit for averages over large fields
                big_grid = RectilinearGrid(arch,
                                           topology = (Periodic, Periodic, Bounded),
                                           size = (256, 256, 128),
                                           x = (0, 2),
                                           y = (0, 2),
                                           z = (0, 2))

                c = CenterField(big_grid)
                c .= 1

                C = Field(Average(c, dims=(1, 2)))

                # Test that the mean consistently returns 1 at every z for many evaluations
                results = [all(interior(mean!(C, C.operand)) .== 1) for i = 1:10] # warm up...
                results = [all(interior(mean!(C, C.operand)) .== 1) for i = 1:10] # the real deal
                @test mean(results) == 1.0              
            end
        end

        @testset "Allocating reductions [$arch_str]" begin
            @info "  Testing allocating reductions"
            
            # Mean
            @test Txyz[1, 1, 1] == mean(T)
            @test interior(Txy) == interior(mean(T, dims=(1, 2)))
            @test interior(Tx) == interior(mean(T, dims=1))

            @test wxyz[1, 1, 1] == mean(w)
            @test interior(wxy) == interior(mean(w, dims=(1, 2)))
            @test interior(wx) == interior(mean(w, dims=1))

            # Maximum and minimum
            @test maximum(T) == maximum(interior(T))
            @test minimum(T) == minimum(interior(T))

            for dims in (1, (1, 2))
                @test interior(minimum(T; dims)) == minimum(interior(T); dims)
                @test interior(minimum(T; dims)) == minimum(interior(T); dims)
            end
        end

        @testset "Conditional computation of averaged Fields [$(typeof(arch))]" begin
            @info "  Testing conditional computation of averaged Fields [$(typeof(arch))]"
            for FT in float_types
                grid = RectilinearGrid(arch, FT, size=(2, 2, 2), extent=(1, 1, 1))
                c = CenterField(grid)

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
