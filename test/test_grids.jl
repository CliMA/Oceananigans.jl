using Oceananigans.Grids: total_extent

#####
##### Grid utilities and such
#####

function test_xnode_ynode_znode_are_correct(FT, N=3)

    grid = RegularCartesianGrid(FT, size=(N, N, N), x=(0, π), y=(0, π), z=(0, π),
                                topology=(Periodic, Periodic, Bounded))

    @test xnode(Cell, 2, grid) ≈ FT(π/2)
    @test ynode(Cell, 2, grid) ≈ FT(π/2)
    @test znode(Cell, 2, grid) ≈ FT(π/2)

    @test xnode(Face, 2, grid) ≈ FT(π/3)
    @test ynode(Face, 2, grid) ≈ FT(π/3)
    @test znode(Face, 2, grid) ≈ FT(π/3)

    @test xC(2, grid) == xnode(Cell, 2, grid)
    @test yC(2, grid) == ynode(Cell, 2, grid)
    @test zC(2, grid) == znode(Cell, 2, grid)

    @test xF(2, grid) == xnode(Face, 2, grid)
    @test yF(2, grid) == ynode(Face, 2, grid)
    @test zF(2, grid) == znode(Face, 2, grid)

    return nothing
end

#####
##### Regular Cartesian grids
#####

function regular_cartesian_correct_size(FT)
    grid = RegularCartesianGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π))

    # Checking ≈ as the grid could be storing Float32 values.
    return (grid.Nx ≈ 4  && grid.Ny ≈ 6  && grid.Nz ≈ 8 &&
            grid.Lx ≈ 2π && grid.Ly ≈ 4π && grid.Lz ≈ 9π)
end

function regular_cartesian_correct_extent(FT)
    grid = RegularCartesianGrid(FT, size=(4, 6, 8), x=(1, 2), y=(π, 3π), z=(0, 4))
    return (grid.Lx ≈ 1 && grid.Ly ≈ 2π  && grid.Lz ≈ 4)
end

function regular_cartesian_correct_coordinate_lengths(FT)
    grid = RegularCartesianGrid(FT, size=(2, 3, 4), extent=(1, 1, 1), halo=(1, 1, 1),
                                topology=(Periodic, Bounded, Flat))

    return (
            length(grid.xC) == 4 &&
            length(grid.yC) == 5 &&
            length(grid.zC) == 6 &&
            length(grid.xF) == 4 &&
            length(grid.yF) == 6 &&
            length(grid.zF) == 6
           )
end

function regular_cartesian_correct_halo_size(FT)
    grid = RegularCartesianGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π), halo=(1, 2, 3))
    return (grid.Hx == 1  && grid.Hy == 2  && grid.Hz == 3)
end

function regular_cartesian_correct_halo_faces(FT)
    N, H, L = 4, 1, 2.0
    Δ = L / N
    grid = RegularCartesianGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))
    return grid.xF[0] == - H * Δ && grid.yF[0] == - H * Δ && grid.zF[0] == - H * Δ
end

function regular_cartesian_correct_first_cells(FT)
    N, H, L = 4, 1, 4.0
    Δ = L / N
    grid = RegularCartesianGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))
    return grid.xC[1] == Δ/2 && grid.yC[1] == Δ/2 && grid.zC[1] == Δ/2
end

function regular_cartesian_correct_end_faces(FT)
    N, L = 4, 2.0
    Δ = L / N
    grid = RegularCartesianGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(1, 1, 1),
                                topology=(Periodic, Bounded, Flat))
    return grid.xF[end] == L && grid.yF[end] == L + Δ && grid.zF[end] == L
end

function regular_cartesian_ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    Hx, Hy, Hz = 1, 2, 1

    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(Hx, Hy, Hz),
                                topology=(Bounded, Bounded, Bounded))

    return (length(grid.xC) == Nx + 2Hx && length(grid.xF) == Nx + 1 + 2Hx &&
            length(grid.yC) == Ny + 2Hy && length(grid.yF) == Ny + 1 + 2Hy &&
            length(grid.zC) == Nz + 2Hz && length(grid.zF) == Nz + 1 + 2Hz)
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function regular_cartesian_no_roundoff_error_in_ranges(FT)
    Nx, Ny, Nz, Hz = 1, 1, 64, 1
    grid = RegularCartesianGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, π/2), halo=(1, 1, Hz))
    return length(grid.zC) == Nz + 2Hz && length(grid.zF) == Nz + 1 + 2Hz
end

function regular_cartesian_grid_properties_are_same_type(FT)
    grid = RegularCartesianGrid(FT, size=(10, 10, 10), extent=(1, 1//7, 2π))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy, grid.Δz], FT)) &&
           all(eltype.([grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
end

#####
##### Vertically stretched grids
#####

function correct_constant_grid_spacings(FT)
    grid = VerticallyStretchedCartesianGrid(FT, size=(16, 16, 16), x=(0, 1), y=(0, 1), zF=collect(0:16))
    return all(grid.ΔzF .== 1) && all(grid.ΔzC .== 1)
end

function correct_quadratic_grid_spacings(FT)
    Nx = Ny = Nz = 16
    grid = VerticallyStretchedCartesianGrid(FT, size=(Nx, Ny, Nz),
                                            x=(0, 1), y=(0, 1), zF=collect(0:Nz).^2)

     zF(k) = (k-1)^2
     zC(k) = (k^2 + (k-1)^2) / 2
    ΔzF(k) = k^2 - (k-1)^2
    ΔzC(k) = 2k - 2

     zF_is_correct = all(isapprox.(  grid.zF[1:Nz+1],  zF.(1:Nz+1) ))
     zC_is_correct = all(isapprox.(  grid.zC[1:Nz],    zC.(1:Nz)   ))
    ΔzF_is_correct = all(isapprox.( grid.ΔzF[1:Nz],   ΔzF.(1:Nz)   ))

    # Note that ΔzC[1, 1, 1] involves a halo point, which is not directly determined by
    # the user-supplied zF
    ΔzC_is_correct = all(isapprox.( grid.ΔzC[2:Nz-1], ΔzC.(2:Nz-1) ))

    return zF_is_correct && zC_is_correct && ΔzF_is_correct && ΔzC_is_correct
end

function correct_tanh_grid_spacings(FT)
    Nx = Ny = Nz = 16

    S = 3  # Stretching factor

    zF(k) = tanh(S * (2 * (k - 1) / Nz - 1)) / tanh(S)

    grid = VerticallyStretchedCartesianGrid(FT, size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), zF=zF)


     zC(k) = (zF(k) + zF(k+1)) / 2
    ΔzF(k) = zF(k+1) - zF(k)
    ΔzC(k) = zC(k) - zC(k-1)

     zF_is_correct = all(isapprox.(  grid.zF[1:Nz+1],  zF.(1:Nz+1) ))
     zC_is_correct = all(isapprox.(  grid.zC[1:Nz],    zC.(1:Nz)   ))
    ΔzF_is_correct = all(isapprox.( grid.ΔzF[1:Nz],   ΔzF.(1:Nz)   ))

    # See correct_quadratic_grid_spacings for an explanation of this test component
    ΔzC_is_correct = all(isapprox.( grid.ΔzC[2:Nz-1], ΔzC.(2:Nz-1) ))

   return zF_is_correct && zC_is_correct && ΔzF_is_correct && ΔzC_is_correct
end

function vertically_stretched_grid_properties_are_same_type(FT)
    Nx, Ny, Nz = 16, 16, 16
    grid = VerticallyStretchedCartesianGrid(FT, size=(16, 16, 16), x=(0,1), y=(0,1), zF=collect(0:16))
    return all(isa.([grid.Lx, grid.Ly, grid.Lz, grid.Δx, grid.Δy], FT)) &&
           all(eltype.([grid.ΔzF, grid.ΔzC, grid.xF, grid.yF, grid.zF, grid.xC, grid.yC, grid.zC]) .== FT)
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

    @testset "Regular Cartesian grid" begin
        @info "  Testing regular Cartesian grid..."

        @testset "Grid initialization" begin
            @info "    Testing grid initialization..."

            for FT in float_types
                @test regular_cartesian_correct_size(FT)
                @test regular_cartesian_correct_extent(FT)
                @test regular_cartesian_correct_coordinate_lengths(FT)
                @test regular_cartesian_correct_halo_size(FT)
                @test regular_cartesian_correct_halo_faces(FT)
                @test regular_cartesian_correct_first_cells(FT)
                @test regular_cartesian_correct_end_faces(FT)
                @test regular_cartesian_ranges_have_correct_length(FT)
                @test regular_cartesian_no_roundoff_error_in_ranges(FT)
                @test regular_cartesian_grid_properties_are_same_type(FT)
            end
        end

        @testset "Grid dimensions" begin
            @info "    Testing grid constructor errors..."

            for FT in float_types
                @test isbitstype(typeof(RegularCartesianGrid(FT, size=(16, 16, 16), extent=(1, 1, 1))))

                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32,), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 64), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32, 16), extent=(1, 1, 1))

                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32.0), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(20.1, 32, 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, nothing, 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, "32", 32), extent=(1, 1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32), extent=(1, nothing, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32), extent=(1, "1", 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1.0, 1, 1))

                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=2)
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), y=[1, 2])
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), z=(-π, π))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=1, y=2, z=3)
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(0, 1), y=(0, 2), z=4)
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=("0", "1"))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=(1, 2, 3))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(1, 0), y=(1//7, 5//7), z=(1, 2))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1))
                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))

                @test_throws ArgumentError RegularCartesianGrid(FT, size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Flux))
            end
        end
    end

    @testset "Vertically stretched Cartesian grid" begin
        @info "  Testing vertically stretched Cartesian grid..."

        @testset "Grid initialization" begin
            @info "    Testing grid initialization..."

            for FT in float_types
                @test correct_constant_grid_spacings(FT)
                @test correct_quadratic_grid_spacings(FT)
                @test correct_tanh_grid_spacings(FT)
                @test vertically_stretched_grid_properties_are_same_type(FT)
            end
        end
    end
end
