include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.Architectures: arch_array
using Oceananigans.AbstractOperations: BinaryOperation
using Oceananigans.Fields: ReducedField, CenterField, ZFaceField, compute_at!, @compute
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: halo_size

trilinear(x, y, z) = x + y + z

@testset "Fields computed by Reduction" begin
    @info "Testing Fields computed by reductions..."

    for arch in archs
        arch_str = string(typeof(arch))

        regular_grid = RectilinearGrid(arch, size = (2, 2, 2),
                                       x = (0, 2), y = (0, 2), z = (0, 2),
                                       topology = (Periodic, Periodic, Bounded))

        xy_regular_grid = RectilinearGrid(arch, size=(2, 2, 2),
                                          x=(0, 2), y=(0, 2), z=[0, 1, 2],
                                          topology = (Periodic, Periodic, Bounded))

        @testset "Averaged fields [$arch_str]" begin
            @info "  Testing averaged Fields [$arch_str]"

            for grid in (regular_grid, xy_regular_grid)

                Nx, Ny, Nz = size(grid)

                w = ZFaceField(grid)
                T = CenterField(grid)
                ζ = Field{Face, Face, Face}(grid)

                set!(T, trilinear)
                set!(w, trilinear)
                set!(ζ, trilinear)

                @compute Txyz = Field(Average(T, dims=(1, 2, 3)))

                # Note: halo regions must be *filled* prior to computing an average
                # if the average within halo regions is to be correct.
                fill_halo_regions!(T, arch)
                @compute Txy = Field(Average(T, dims=(1, 2)))

                fill_halo_regions!(T, arch)
                @compute Tx = Field(Average(T, dims=1))

                @compute wxyz = Field(Average(w, dims=(1, 2, 3)))
                @compute wxy = Field(Average(w, dims=(1, 2)))
                @compute wx = Field(Average(w, dims=1))

                @compute ζxyz = Field(Average(ζ, dims=(1, 2, 3)))
                @compute ζxy = Field(Average(ζ, dims=(1, 2)))
                @compute ζx = Field(Average(ζ, dims=1))

                for T′ in (Tx, Txy)
                    @test T′.operand.operand === T
                end
                
                for w′ in (wx, wxy)
                    @test w′.operand.operand === w
                end

                for ζ′ in (ζx, ζxy)
                    @test ζ′.operand.operand === ζ
                end

                for f in (wx, wxy, Tx, Txy, ζx, ζxy)
                    @test f.operand isa Reduction
                    @test f.operand.reduce! === mean!
                end

                @test Txyz.operand isa Reduction
                @test wxyz.operand isa Reduction
                @test ζxyz.operand isa Reduction

                # Different behavior for regular grid z vs not.
                if grid === regular_grid
                    @test Txyz.operand.reduce! === mean!
                    @test wxyz.operand.reduce! === mean!
                    @test Txyz.operand.operand === T
                    @test wxyz.operand.operand === w
                else
                    @test Txyz.operand.reduce! === sum!
                    @test wxyz.operand.reduce! === sum!
                    @test Txyz.operand.operand isa BinaryOperation
                    @test wxyz.operand.operand isa BinaryOperation
                end

                @test Tx.operand.dims === tuple(1)
                @test wx.operand.dims === tuple(1)
                @test Txy.operand.dims === (1, 2)
                @test wxy.operand.dims === (1, 2)
                @test Txyz.operand.dims === (1, 2, 3)
                @test wxyz.operand.dims === (1, 2, 3)

                @test Txyz[1, 1, 1] ≈ 3
                @test Array(interior(Txy))[1, 1, :] ≈ [2.5, 3.5]
                @test Array(interior(Tx))[1, :, :] ≈ [[2, 3] [3, 4]]
                @test wxyz[1, 1, 1] ≈ 3
                @test Array(interior(wxy))[1, 1, :] ≈ [2, 3, 4]
                @test Array(interior(wx))[1, :, :] ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
                
                @compute Txyz = Field(Average(T, condition=T.>3))
                @compute Txy = Field(Average(T, dims=(1, 2), condition=T.>3))
                @compute Tx = Field(Average(T, dims=1, condition=T.>2))
                @test Txyz[1,1,1] ≈ 3.75
                @test Array(interior(Txy))[1, 1, :] ≈ [3.5, 11.5/3]
                @test Array(interior(Tx))[1, :, :] ≈ [[2.5, 3] [3, 4]]

                @compute wxyz = Field(Average(w, condition=w.>3))
                @compute wxy = Field(Average(w, dims=(1, 2), condition=w.>2))
                @compute wx = Field(Average(w, dims=1, condition=w.>1))
                @test wxyz[1,1,1] ≈ 4.25
                @test Array(interior(wxy))[1, 1, :] ≈ [3, 10/3, 4]
                @test Array(interior(wx))[1, :, :] ≈ [[2, 2.5] [2.5, 3.5] [3.5, 4.5]]
            end
            
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
            # Warm up
            for i = 1:10
                sum!(C, C.operand.operand)
            end

            results = []
            for i = 1:10
                mean!(C, C.operand.operand)
                push!(results, all(interior(C) .== 1))
            end

            @test mean(results) == 1.0              
        end

        @testset "Allocating reductions [$arch_str]" begin
            @info "  Testing allocating reductions"

            grid = RectilinearGrid(arch, size = (2, 2, 2),
                                   x = (0, 2), y = (0, 2), z = (0, 2),
                                   topology = (Periodic, Periodic, Bounded))

            w = ZFaceField(grid)
            T = CenterField(grid)
            set!(T, trilinear)
            set!(w, trilinear)
            fill_halo_regions!(T, arch)
            fill_halo_regions!(w, arch)

            @compute Txyz = Field(Average(T, dims=(1, 2, 3)))
            @compute Txy = Field(Average(T, dims=(1, 2)))
            @compute Tx = Field(Average(T, dims=1))

            @compute wxyz = Field(Average(w, dims=(1, 2, 3)))
            @compute wxy = Field(Average(w, dims=(1, 2)))
            @compute wx = Field(Average(w, dims=1))

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
                    C = Field(Average(c, dims=dims))

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
