using Statistics
using Oceananigans.Architectures: arch_array
using Oceananigans.Fields: ReducedField, CenterField, ZFaceField, compute_at!, @compute
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: halo_size

include("utils_for_runtests.jl")

archs = test_architectures()

"""
    correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz)

Test that the ReducedField at `loc`ation on `arch`itecture and `grid`
and reduced along `dims` has size `(Tx, Ty, Tz)`.
"""
correct_reduced_field_size(loc, arch, grid, dims, Tx, Ty, Tz) =
    size(parent(ReducedField(loc, arch, grid; dims=dims))) == (Tx, Ty, Tz)

function correct_reduced_field_value_was_set(arch, grid, loc, dims, val::Number)
    f = ReducedField(loc, arch, grid; dims=dims)
    set!(f, val)
    return all(interior(f) .≈ val * arch_array(arch, ones(size(f))))
end

@testset "AbstractReducedFields" begin
    @info "Testing AbstractReducedFields..."

    N = (4, 6, 8)
    L = (2π, 3π, 5π)
    H = (1, 1, 1)

    @testset "ReducedField initialization" begin
        @info "  Testing ReducedField initialization..."
        for arch in archs, FT in float_types

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Bounded, Bounded, Bounded))

            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Face,   Center, Center), arch, grid, 1,         1,               N[2] + 2 * H[2],     N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Face,   Center), arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Face,   Face),   arch, grid, 1,         1,               N[2] + 2 * H[2] + 1, N[3] + 2 * H[3] + 1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 2,         N[1] + 2 * H[1], 1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, 3,         N[1] + 2 * H[1], N[2] + 2 * H[2],     1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (1, 2),    1,               1,                   N[3] + 2 * H[3])
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (2, 3),    N[1] + 2 * H[1], 1,                   1)
            @test correct_reduced_field_size((Center, Center, Center), arch, grid, (1, 2, 3), 1,               1,                   1)
        end
    end

    reduced_dims = (1, 2, 3, (1, 2), (2, 3), (1, 3), (1, 2, 3))

    int_vals = Any[0, Int8(-1), Int16(2), Int32(-3), Int64(4)]
    uint_vals = Any[6, UInt8(7), UInt16(8), UInt32(9), UInt64(10)]
    float_vals = Any[0.0, -0.0, 6e-34, 1.0f10]
    rational_vals = Any[1//11, -23//7]
    other_vals = Any[π]
    vals = vcat(int_vals, uint_vals, float_vals, rational_vals, other_vals)

    @testset "Setting ReducedFields" begin
        @info "  Testing ReducedField setting..."
        for arch in archs, FT in float_types

            grid = RectilinearGrid(arch, FT, size=N, extent=L, halo=H, topology=(Periodic, Periodic, Bounded))

            for dims in reduced_dims, val in vals
                @test correct_reduced_field_value_was_set(arch, grid, (Center, Center, Center), dims, val)
            end

            for dims in reduced_dims
                reduced_field = ReducedField((Center, Center, Center), arch, grid, dims=dims)
                sz = size(reduced_field)
                A = rand(FT, sz...)
                set!(reduced_field, A)
                
                @test reduced_field[1, 1, 1] == A[1, 1, 1]

                fill_halo_regions!(reduced_field, arch)

                # No-flux boundary conditions at top and bottom
                @test reduced_field[1, 1, 0] == A[1, 1, 1]
                @test reduced_field[1, 1, grid.Nz+1] == A[1, 1, end]

                # Periodic boundary conditions in the horizontal directions
                @test reduced_field[1, 0, 1] == A[1, end, 1]
                @test reduced_field[1, grid.Ny+1, 1] == A[1, 1, 1]

                @test reduced_field[0, 1, 1] == A[end, 1, 1]
                @test reduced_field[grid.Nx+1, 1, 1] == A[1, 1, 1]
            end
        end
    end

    for arch in archs
        arch_str = string(typeof(arch))

        grid = RectilinearGrid(arch, topology = (Periodic, Periodic, Bounded),
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

        @testset "Averaged fields [$arch_str]" begin
            @info "  Testing AveragedFields [$arch_str]"
            for FT in float_types
                
                @test Txyz[1, 1, 1] ≈ 3

                @test Array(interior(Txy))[1, 1, :] ≈ [2.5, 3.5]
                @test Array(interior(Tx))[1, :, :] ≈ [[2, 3] [3, 4]]

                @test wxyz[1, 1, 1] ≈ 3

                @test Array(interior(wxy))[1, 1, :] ≈ [2, 3, 4]
                @test Array(interior(wx))[1, :, :] ≈ [[1.5, 2.5] [2.5, 3.5] [3.5, 4.5]]
                
                # Test whether a race condition gets hit for averages over large fields
                big_grid = RectilinearGrid(arch, topology = (Periodic, Periodic, Bounded),
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

        @testset "Conditional computation of AveragedFields [$(typeof(arch))]" begin
            @info "  Testing conditional computation of AveragedFields [$(typeof(arch))]"
            for FT in float_types
                grid = RectilinearGrid(arch, FT, size=(2, 2, 2), extent=(1, 1, 1))
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
