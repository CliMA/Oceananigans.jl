using Oceananigans.Grids: total_extent

#####
##### Grid utilities and such
#####

function test_xnode_ynode_znode_are_correct(FT, N=3)

    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, π), y=(0, π), z=(0, π),
                                topology=(Periodic, Periodic, Bounded))

    @test xnode(Center, 2, grid) ≈ FT(π/2)
    @test ynode(Center, 2, grid) ≈ FT(π/2)
    @test znode(Center, 2, grid) ≈ FT(π/2)

    @test xnode(Face, 2, grid) ≈ FT(π/3)
    @test ynode(Face, 2, grid) ≈ FT(π/3)
    @test znode(Face, 2, grid) ≈ FT(π/3)

    @test xC(2, grid) == xnode(Center, 2, grid)
    @test yC(2, grid) == ynode(Center, 2, grid)
    @test zC(2, grid) == znode(Center, 2, grid)

    @test xF(2, grid) == xnode(Face, 2, grid)
    @test yF(2, grid) == ynode(Face, 2, grid)
    @test zF(2, grid) == znode(Face, 2, grid)

    return nothing
end

#####
##### Regular rectilinear grids
#####

function regular_rectilinear_correct_size(FT)
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    return (grid.Nx ≈ 4  && grid.Ny ≈ 6  && grid.Nz ≈ 8 &&
            grid.Lx ≈ 2π && grid.Ly ≈ 4π && grid.Lz ≈ 9π)
end

function regular_rectilinear_correct_extent(FT)
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), x=(1, 2), y=(π, 3π), z=(0, 4))
    return (grid.Lx ≈ 1 && grid.Ly ≈ 2π  && grid.Lz ≈ 4)
end

function regular_rectilinear_correct_coordinate_lengths(FT)
    grid = RegularRectilinearGrid(FT, size=(2, 3, 4), extent=(1, 1, 1), halo=(1, 1, 1),
                                topology=(Periodic, Bounded, Bounded))

    return (
            length(grid.xC) == 4 &&
            length(grid.yC) == 5 &&
            length(grid.zC) == 6 &&
            length(grid.xF) == 4 &&
            length(grid.yF) == 6 &&
            length(grid.zF) == 7
           )
end

function regular_rectilinear_correct_halo_size(FT)
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π), halo=(1, 2, 3))
    return (grid.Hx == 1  && grid.Hy == 2  && grid.Hz == 3)
end

function regular_rectilinear_correct_halo_faces(FT)
    N, H, L = 4, 1, 2.0
    Δ = L / N
    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))
    return grid.xF[0] == - H * Δ && grid.yF[0] == - H * Δ && grid.zF[0] == - H * Δ
end

function regular_rectilinear_correct_first_cells(FT)
    N, H, L = 4, 1, 4.0
    Δ = L / N
    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))
    return grid.xC[1] == Δ/2 && grid.yC[1] == Δ/2 && grid.zC[1] == Δ/2
end

function regular_rectilinear_correct_end_faces(FT)
    N, L = 4, 2.0
    Δ = L / N
    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(1, 1, 1),
                                topology=(Periodic, Bounded, Bounded))
    return grid.xF[end] == L && grid.yF[end] == L + Δ && grid.zF[end] == L + Δ
end

function regular_rectilinear_ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    Hx, Hy, Hz = 1, 2, 1

    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(Hx, Hy, Hz),
                                topology=(Bounded, Bounded, Bounded))

    return (length(grid.xC) == Nx + 2Hx && length(grid.xF) == Nx + 1 + 2Hx &&
            length(grid.yC) == Ny + 2Hy && length(grid.yF) == Ny + 1 + 2Hy &&
            length(grid.zC) == Nz + 2Hz && length(grid.zF) == Nz + 1 + 2Hz)
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function regular_rectilinear_no_roundoff_error_in_ranges(FT)
    Nx, Ny, Nz, Hz = 1, 1, 64, 1
    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, π/2), halo=(1, 1, Hz))
    return length(grid.zC) == Nz + 2Hz && length(grid.zF) == Nz + 1 + 2Hz
end

function regular_rectilinear_grid_properties_are_same_type(FT)
    grid = RegularRectilinearGrid(FT, size=(10, 10, 10), extent=(1, 1//7, 2π))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy, grid.Δz], FT)) &&
           all(eltype.([grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
end

#####
##### Vertically stretched grids
#####

function vertically_stretched_grid_properties_are_same_type(FT, arch)
    grid = VerticallyStretchedRectilinearGrid(FT, architecture=arch, size=(1, 1, 16), x=(0,1), y=(0,1), zF=collect(0:16))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy], FT)) &&
           all(eltype.([grid.ΔzF, grid.ΔzC, grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
end

function run_architecturally_correct_stretched_grid_tests(FT, arch, zF)
    grid = VerticallyStretchedRectilinearGrid(FT, architecture=arch, size=(1, 1, length(zF)-1), x=(0, 1), y=(0, 1), zF=zF)

    ArrayType = array_type(arch)
    @test grid.zF  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.zC  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.ΔzF isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.ΔzC isa OffsetArray{FT, 1, <:ArrayType}

    return nothing
end

function run_correct_constant_grid_spacings_tests(FT, Nz)
    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz))
    @test all(grid.ΔzF .== 1)
    @test all(grid.ΔzC .== 1)
    return nothing
end

function run_correct_quadratic_grid_spacings_tests(FT, Nz)
    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz).^2)

     zF(k) = (k-1)^2
     zC(k) = (k^2 + (k-1)^2) / 2
    ΔzF(k) = k^2 - (k-1)^2
    ΔzC(k) = zC(k+1) - zC(k)

    @test all(isapprox.(  grid.zF[1:Nz+1],  zF.(1:Nz+1) ))
    @test all(isapprox.(  grid.zC[1:Nz],    zC.(1:Nz)   ))
    @test all(isapprox.( grid.ΔzF[1:Nz],   ΔzF.(1:Nz)   ))

    # Note that ΔzC[1] involves a halo point, which is not directly determined by
    # the user-supplied zF
    @test all(isapprox.( grid.ΔzC[2:Nz-1], ΔzC.(2:Nz-1) ))

    return nothing
end

function run_correct_tanh_grid_spacings_tests(FT, Nz)
    S = 3  # Stretching factor
    zF(k) = tanh(S * (2 * (k - 1) / Nz - 1)) / tanh(S)

    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=zF)

     zC(k) = (zF(k) + zF(k+1)) / 2
    ΔzF(k) = zF(k+1) - zF(k)
    ΔzC(k) = zC(k+1) - zC(k)

    @test all(isapprox.(  grid.zF[1:Nz+1],  zF.(1:Nz+1) ))
    @test all(isapprox.(  grid.zC[1:Nz],    zC.(1:Nz)   ))
    @test all(isapprox.( grid.ΔzF[1:Nz],   ΔzF.(1:Nz)   ))

    # Note that ΔzC[1] involves a halo point, which is not directly determined by
    # the user-supplied zF
    @test all(isapprox.( grid.ΔzC[2:Nz-1], ΔzC.(2:Nz-1) ))

   return nothing
end

function flat_size_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RegularRectilinearGrid(FT; size=size, topology=topology, extent=extent)
    return grid.Nx, grid.Ny, grid.Nz
end

function flat_halo_regular_rectilinear_grid(FT; topology, size, halo, extent)
    grid = RegularRectilinearGrid(FT; size=size, halo=halo, topology=topology, extent=extent)
    return grid.Hx, grid.Hy, grid.Hz
end

function flat_extent_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RegularRectilinearGrid(FT; size=size, topology=topology, extent=extent)
    return grid.Lx, grid.Ly, grid.Lz
end


#####
##### Test the tests
#####

@testset "Grids" begin
    @info "Testing grids..."

    @testset "Grid utils" begin
        @info "  Testing grid utilities..."
        @test total_extent(Periodic, 1, 0.2, 1.0) == 1.2
        @test total_extent(Bounded, 1, 0.2, 1.0) == 1.4
        for FT in float_types
            test_xnode_ynode_znode_are_correct(FT)
        end
    end

    @testset "Regular rectilinear grid" begin
        @info "  Testing regular rectilinear grid..."

        @testset "Grid initialization" begin
            @info "    Testing grid initialization..."

            for FT in float_types
                @test regular_rectilinear_correct_size(FT)
                @test regular_rectilinear_correct_extent(FT)
                @test regular_rectilinear_correct_coordinate_lengths(FT)
                @test regular_rectilinear_correct_halo_size(FT)
                @test regular_rectilinear_correct_halo_faces(FT)
                @test regular_rectilinear_correct_first_cells(FT)
                @test regular_rectilinear_correct_end_faces(FT)
                @test regular_rectilinear_ranges_have_correct_length(FT)
                @test regular_rectilinear_no_roundoff_error_in_ranges(FT)
                @test regular_rectilinear_grid_properties_are_same_type(FT)
            end
        end

        @testset "Grid dimensions" begin
            @info "    Testing grid constructor errors..."

            for FT in float_types
                @test isbitstype(typeof(RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, 1))))

                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32,), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 64), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32, 16), extent=(1, 1, 1))

                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32.0), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(20.1, 32, 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, nothing, 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, "32", 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32), extent=(1, nothing, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32), extent=(1, "1", 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1.0, 1, 1))

                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=2)
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), y=[1, 2])
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), z=(-π, π))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=1, y=2, z=3)
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(0, 2), z=4)
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=("0", "1"))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=(1, 2, 3))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(1, 0), y=(1//7, 5//7), z=(1, 2))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))

                @test_throws ArgumentError RegularRectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Flux))

                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Flat, Periodic, Periodic), size=(16, 16, 16), extent=1)
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Periodic, Flat, Periodic), size=(16, 16, 16), extent=(1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=(16, 16, 16), extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=(16, 16),     extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=16,           extent=(1, 1, 1))

                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Periodic, Flat, Flat), size=16, extent=(1, 1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Flat, Periodic, Flat), size=16, extent=(1, 1))
                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Flat, Flat, Periodic), size=(16, 16), extent=1)

                @test_throws ArgumentError RegularRectilinearGrid(FT, topology=(Flat, Flat, Flat), size=16, extent=1)
            end
        end

        @testset "Grids with flat dimensions" begin
            @info "    Testing construction of grids with Flat dimensions..."

            for FT in float_types
                @test flat_size_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) === (1, 2, 3)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Bounded),  size=(2, 3), extent=(1, 1)) === (2, 1, 3)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Periodic, Bounded, Flat),  size=(2, 3), extent=(1, 1)) === (2, 3, 1)

                @test flat_size_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) === (1, 2, 3)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Bounded),  size=(2, 3), extent=(1, 1)) === (2, 1, 3)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Periodic, Bounded, Flat),  size=(2, 3), extent=(1, 1)) === (2, 3, 1)

                @test flat_size_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Flat), size=2, extent=1) === (2, 1, 1)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Flat), size=2, extent=1) === (1, 2, 1)
                @test flat_size_regular_rectilinear_grid(FT; topology=(Flat, Flat, Bounded),  size=2, extent=1) === (1, 1, 2)

                @test flat_size_regular_rectilinear_grid(FT; topology=(Flat, Flat, Flat), size=(), extent=()) === (1, 1, 1)

                @test flat_halo_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Periodic), size=(1, 1), extent=(1, 1), halo=nothing) === (0, 1, 1)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Bounded),  size=(1, 1), extent=(1, 1), halo=nothing) === (1, 0, 1)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Periodic, Bounded, Flat),  size=(1, 1), extent=(1, 1), halo=nothing) === (1, 1, 0)

                @test flat_halo_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Periodic), size=(1, 1), extent=(1, 1), halo=(2, 3)) === (0, 2, 3)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Bounded),  size=(1, 1), extent=(1, 1), halo=(2, 3)) === (2, 0, 3)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Periodic, Bounded, Flat),  size=(1, 1), extent=(1, 1), halo=(2, 3)) === (2, 3, 0)

                @test flat_halo_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Flat), size=1, extent=1, halo=2) === (2, 0, 0)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Flat), size=1, extent=1, halo=2) === (0, 2, 0)
                @test flat_halo_regular_rectilinear_grid(FT; topology=(Flat, Flat, Bounded),  size=1, extent=1, halo=2) === (0, 0, 2)

                @test flat_halo_regular_rectilinear_grid(FT; topology=(Flat, Flat, Flat), size=(), extent=(), halo=()) === (0, 0, 0)

                @test flat_extent_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) == (0, 1, 1)
                @test flat_extent_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Periodic), size=(2, 3), extent=(1, 1)) == (1, 0, 1)
                @test flat_extent_regular_rectilinear_grid(FT; topology=(Periodic, Periodic, Flat), size=(2, 3), extent=(1, 1)) == (1, 1, 0)

                @test flat_extent_regular_rectilinear_grid(FT; topology=(Periodic, Flat, Flat), size=2, extent=1) == (1, 0, 0)
                @test flat_extent_regular_rectilinear_grid(FT; topology=(Flat, Periodic, Flat), size=2, extent=1) == (0, 1, 0)
                @test flat_extent_regular_rectilinear_grid(FT; topology=(Flat, Flat, Periodic), size=2, extent=1) == (0, 0, 1)

                @test flat_extent_regular_rectilinear_grid(FT; topology=(Flat, Flat, Flat), size=(), extent=()) == (0, 0, 0)
            end
        end

        # Testing show function
        topo = (Periodic, Periodic, Periodic)
        grid = RegularRectilinearGrid(topology=topo, size=(3, 7, 9), x=(0, 1), y=(-π, π), z=(0, 2π))
        show(grid); println();
        @test grid isa RegularRectilinearGrid
    end

    @testset "Vertically stretched rectilinear grid" begin
        @info "  Testing vertically stretched rectilinear grid..."

        for arch in archs, FT in float_types
            @testset "Vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]" begin
                @info "    Testing vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]..."
                @test vertically_stretched_grid_properties_are_same_type(FT, arch)

                zF1 = collect(0:10).^2
                zF2 = [1, 3, 5, 10, 15, 33, 50]
                for zF in [zF1, zF2]
                    run_architecturally_correct_stretched_grid_tests(FT, arch, zF)
                end
            end

            @testset "Vertically stretched rectilinear grid spacings [$(typeof(arch)), $FT]" begin
                @info "    Testing vertically stretched rectilinear grid spacings [$(typeof(arch)), $FT]..."
                for Nz in [16, 17]
                    run_correct_constant_grid_spacings_tests(FT, Nz)
                    run_correct_quadratic_grid_spacings_tests(FT, Nz)
                    run_correct_tanh_grid_spacings_tests(FT, Nz)
                end
            end

            # Testing show function
            Nz = 20
            grid = VerticallyStretchedRectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz).^2)
            show(grid); println();
            @test grid isa VerticallyStretchedRectilinearGrid
        end
    end
end
