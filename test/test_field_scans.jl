include("dependencies_for_runtests.jl")

using Statistics
using Oceananigans.Architectures: on_architecture
using Oceananigans.AbstractOperations: BinaryOperation
using Oceananigans.Fields: ReducedField, CenterField, ZFaceField, compute_at!, @compute, reverse_cumsum!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: halo_size

trilinear(x, y, z) = x + y + z
interior_array(a, i, j, k) = Array(interior(a, i, j, k))

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

        @testset "Averaged and integrated fields [$arch_str]" begin
            @info "  Testing averaged and integrated Fields [$arch_str]"

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
                fill_halo_regions!(T)
                @compute Txy = Field(Average(T, dims=(1, 2)))

                fill_halo_regions!(T)
                @compute Tx = Field(Average(T, dims=1))

                @compute wxyz = Field(Average(w, dims=(1, 2, 3)))
                @compute wxy = Field(Average(w, dims=(1, 2)))
                @compute wx = Field(Average(w, dims=1))

                @compute ζxyz = Field(Average(ζ, dims=(1, 2, 3)))
                @compute ζxy = Field(Average(ζ, dims=(1, 2)))
                @compute ζx = Field(Average(ζ, dims=1))

                @compute Θxyz = Field(Integral(T, dims=(1, 2, 3)))
                @compute Θxy = Field(Integral(T, dims=(1, 2)))
                @compute Θx = Field(Integral(T, dims=1))

                @compute Wxyz = Field(Integral(w, dims=(1, 2, 3)))
                @compute Wxy = Field(Integral(w, dims=(1, 2)))
                @compute Wx = Field(Integral(w, dims=1))

                @compute Zxyz = Field(Integral(ζ, dims=(1, 2, 3)))
                @compute Zxy = Field(Integral(ζ, dims=(1, 2)))
                @compute Zx = Field(Integral(ζ, dims=1))

                @compute Tcx = Field(CumulativeIntegral(T, dims=1))
                @compute Tcy = Field(CumulativeIntegral(T, dims=2))
                @compute Tcz = Field(CumulativeIntegral(T, dims=3))
                @compute wcx = Field(CumulativeIntegral(w, dims=1))
                @compute wcy = Field(CumulativeIntegral(w, dims=2))
                @compute wcz = Field(CumulativeIntegral(w, dims=3))
                @compute ζcx = Field(CumulativeIntegral(ζ, dims=1))
                @compute ζcy = Field(CumulativeIntegral(ζ, dims=2))
                @compute ζcz = Field(CumulativeIntegral(ζ, dims=3))

                @compute Trx = Field(CumulativeIntegral(T, dims=1, reverse=true))
                @compute Try = Field(CumulativeIntegral(T, dims=2, reverse=true))
                @compute Trz = Field(CumulativeIntegral(T, dims=3, reverse=true))
                @compute wrx = Field(CumulativeIntegral(w, dims=1, reverse=true))
                @compute wry = Field(CumulativeIntegral(w, dims=2, reverse=true))
                @compute wrz = Field(CumulativeIntegral(w, dims=3, reverse=true))
                @compute ζrx = Field(CumulativeIntegral(ζ, dims=1, reverse=true))
                @compute ζry = Field(CumulativeIntegral(ζ, dims=2, reverse=true))
                @compute ζrz = Field(CumulativeIntegral(ζ, dims=3, reverse=true))

                for T′ in (Tx, Txy)
                    @test T′.operand.operand === T
                end
                
                for w′ in (wx, wxy)
                    @test w′.operand.operand === w
                end

                for ζ′ in (ζx, ζxy)
                    @test ζ′.operand.operand === ζ
                end

                for f in (wx, wxy, Tx, Txy, ζx, ζxy, Wx, Wxy, Θx, Θxy, Zx, Zxy)
                    @test f.operand isa Reduction
                end

                for f in (Tcx, Tcy, Tcz, Trx, Try, Trz,
                          wcx, wcy, wcz, wrx, wry, wrz,
                          ζcx, ζcy, ζcz, ζrx, ζry, ζrz)
                    @test f.operand isa Accumulation
                end

                for f in (wx, wxy, Tx, Txy, ζx, ζxy)
                    @test f.operand.scan! === mean!
                end

                for f in (wx, wxy, Tx, Txy, ζx, ζxy)
                    @test f.operand.scan! === mean!
                end

                for f in (Tcx, Tcy, Tcz, wcx, wcy, wcz, ζcx, ζcy, ζcz)
                    @test f.operand.scan! === cumsum!
                end

                for f in (Trx, Try, Trz, wrx, wry, wrz, ζrx, ζry, ζrz)
                    @test f.operand.scan! === reverse_cumsum!
                end

                @test Txyz.operand isa Reduction
                @test wxyz.operand isa Reduction
                @test ζxyz.operand isa Reduction

                # Different behavior for regular grid z vs not.
                if grid === regular_grid
                    @test Txyz.operand.scan! === mean!
                    @test wxyz.operand.scan! === mean!
                    @test Txyz.operand.operand === T
                    @test wxyz.operand.operand === w
                else
                    @test Txyz.operand.scan! === sum!
                    @test wxyz.operand.scan! === sum!
                    @test Txyz.operand.operand isa BinaryOperation
                    @test wxyz.operand.operand isa BinaryOperation
                end

                @test Tx.operand.dims === tuple(1)
                @test wx.operand.dims === tuple(1)
                @test Txy.operand.dims === (1, 2)
                @test wxy.operand.dims === (1, 2)
                @test Txyz.operand.dims === (1, 2, 3)
                @test wxyz.operand.dims === (1, 2, 3)

                @test CUDA.@allowscalar Txyz[1, 1, 1] ≈ 3
                @test interior_array(Txy, 1, 1, :) ≈ [2.5, 3.5]
                @test interior_array(Tx, 1, :, :) ≈ [[2, 3] [3, 4]]

                @test CUDA.@allowscalar wxyz[1, 1, 1] ≈ 3
                @test interior_array(wxy, 1, 1, :) ≈ [2, 3, 4]
                @test interior_array(wx, 1, :, :) ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]

                averages_1d  = (Tx, wx, ζx)
                integrals_1d = (Θx, Wx, Zx)

                for (a, i) in zip(averages_1d, integrals_1d)
                    @test interior(i) == 2 .* interior(a)
                end

                averages_2d  = (Txy, wxy, ζxy)
                integrals_2d = (Θxy, Wxy, Zxy)

                for (a, i) in zip(averages_2d, integrals_2d)
                    @test interior(i) == 4 .* interior(a)
                end

                # T(x, y, z) = x + y + z
                # T(0.5, 0.5, z) = [1.5, 2.5]
                @test interior_array(Tcx, :, 1, 1) ≈ [1.5, 4]
                @test interior_array(Tcy, 1, :, 1) ≈ [1.5, 4]
                @test interior_array(Tcz, 1, 1, :) ≈ [1.5, 4]

                @test interior_array(Trx, :, 1, 1) ≈ [4, 2.5]
                @test interior_array(Try, 1, :, 1) ≈ [4, 2.5]
                @test interior_array(Trz, 1, 1, :) ≈ [4, 2.5]

                # w(x, y, z) = x + y + z
                # w(0.5, 0.5, z) = [1, 2, 3]
                # w(x, 0.5, 0) = w(0.5, y, 0) = [1, 2]
                @test interior_array(wcx, :, 1, 1) ≈ [1, 3]
                @test interior_array(wcy, 1, :, 1) ≈ [1, 3]
                @test interior_array(wcz, 1, 1, :) ≈ [1, 3, 6]

                @test interior_array(wrx, :, 1, 1) ≈ [3, 2]
                @test interior_array(wry, 1, :, 1) ≈ [3, 2]
                @test interior_array(wrz, 1, 1, :) ≈ [6, 5, 3]

                @compute Txyz = CUDA.@allowscalar Field(Average(T, condition=T.>3))
                @compute Txy = CUDA.@allowscalar Field(Average(T, dims=(1, 2), condition=T.>3))
                @compute Tx = CUDA.@allowscalar Field(Average(T, dims=1, condition=T.>2))

                @test CUDA.@allowscalar Txyz[1, 1, 1] ≈ 3.75
                @test Array(interior(Txy))[1, 1, :] ≈ [3.5, 11.5/3]
                @test Array(interior(Tx))[1, :, :] ≈ [[2.5, 3] [3, 4]]

                @compute wxyz = CUDA.@allowscalar Field(Average(w, condition=w.>3))
                @compute wxy = CUDA.@allowscalar Field(Average(w, dims=(1, 2), condition=w.>2))
                @compute wx = CUDA.@allowscalar Field(Average(w, dims=1, condition=w.>1))

                @test CUDA.@allowscalar wxyz[1, 1, 1] ≈ 4.25
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
            fill_halo_regions!(T)
            fill_halo_regions!(w)

            @compute Txyz = Field(Average(T, dims=(1, 2, 3)))
            @compute Txy = Field(Average(T, dims=(1, 2)))
            @compute Tx = Field(Average(T, dims=1))

            @compute wxyz = Field(Average(w, dims=(1, 2, 3)))
            @compute wxy = Field(Average(w, dims=(1, 2)))
            @compute wx = Field(Average(w, dims=1))

            # Mean
            @test CUDA.@allowscalar Txyz[1, 1, 1] == mean(T)
            @test interior(Txy) == interior(mean(T, dims=(1, 2)))
            @test interior(Tx) == interior(mean(T, dims=1))

            @test CUDA.@allowscalar wxyz[1, 1, 1] == mean(w)
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

        @testset "Immersed Fields reduction [$(typeof(arch))]" begin
            @info "  Testing reductions of immersed Fields [$(typeof(arch))]"
            underlying_grid = RectilinearGrid(arch, size=(3, 3, 3), extent=(1, 1, 1))
            
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> y < 0.5 ? - 0.6 : 0))
            c = Field((Center, Center, Nothing), grid)

            set!(c, (x, y) -> y)
            @test maximum(c) == grid.yᵃᶜᵃ[1]

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> y < 0.5 ? - 0.6 : -0.4))
            c = Field((Center, Center, Nothing), grid)

            set!(c, (x, y) -> y)
            @test maximum(c) == grid.yᵃᶜᵃ[3]

            underlying_grid = RectilinearGrid(arch, size = (1, 1, 8), extent=(1, 1, 1))

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> -3/4))
            c = Field((Center, Center, Center), grid)

            set!(c, (x, y, z) -> -z)
            @test maximum(c) == Array(interior(c))[1, 1, 3]
            c_condition = interior(c) .< 0.5
            avg_c_smaller_than_½ = Array(interior(compute!(Field(Average(c, condition=c_condition)))))
            @test avg_c_smaller_than_½[1, 1, 1] == 0.25

            zᶜᶜᶜ = KernelFunctionOperation{Center, Center, Center}(znode, grid, Center(), Center(), Center())
            ci = Array(interior(c)) # transfer to CPU
            bottom_half_average_manual = (ci[1, 1, 3] + ci[1, 1, 4]) / 2
            bottom_half_average = Average(c; condition=(zᶜᶜᶜ .< -1/2))
            bottom_half_average_field = Field(bottom_half_average)
            compute!(bottom_half_average_field)
            bottom_half_average_array = Array(interior(bottom_half_average_field))
            @test bottom_half_average_array[1, 1, 1] == bottom_half_average_manual
        end
    end
end
