using Oceananigans.Grids: total_extent, halo_size

#####
##### Regular rectilinear grids
#####

function test_regular_rectilinear_correct_size(FT)
    grid = RectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π))

    @test grid.Nx == 4
    @test grid.Ny == 6
    @test grid.Nz == 8

    # Checking ≈ as the grid could be storing Float32 values.
    @test grid.Lx ≈ 2π
    @test grid.Ly ≈ 4π
    @test grid.Lz ≈ 9π

    return nothing
end

function test_regular_rectilinear_correct_extent(FT)
    grid = RectilinearGrid(FT, size=(4, 6, 8), x=(1, 2), y=(π, 3π), z=(0, 4))

    @test grid.Lx ≈ 1
    @test grid.Ly ≈ 2π
    @test grid.Lz ≈ 4

    return nothing
end

function test_regular_rectilinear_correct_coordinate_lengths(FT)
    grid = RectilinearGrid(FT, size=(2, 3, 4), extent=(1, 1, 1), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    @test length(grid.xᶜᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶜᵃ) == Ny + 2Hy
    @test length(grid.zᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.xᶠᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶠᵃ) == Ny + 2Hy + 1
    @test length(grid.zᵃᵃᶠ) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_correct_halo_size(FT)
    grid = RectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π), halo=(1, 2, 3))

    @test grid.Hx == 1
    @test grid.Hy == 2
    @test grid.Hz == 3

    return nothing
end

function test_regular_rectilinear_correct_halo_faces(FT)
    N = 4
    H = 1
    L = 2.0
    Δ = L / N

    topo = (Periodic, Bounded, Bounded)
    grid = RectilinearGrid(FT, topology=topo, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xᶠᵃᵃ[0] == - H * Δ
    @test grid.yᵃᶠᵃ[0] == - H * Δ
    @test grid.zᵃᵃᶠ[0] == - H * Δ

    @test grid.xᶠᵃᵃ[N+1] == L  # Periodic
    @test grid.yᵃᶠᵃ[N+2] == L + H * Δ
    @test grid.zᵃᵃᶠ[N+2] == L + H * Δ

    return nothing
end

function test_regular_rectilinear_correct_first_cells(FT)
    N = 4
    H = 1
    L = 4.0
    Δ = L / N

    grid = RectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xᶜᵃᵃ[1] == Δ/2
    @test grid.yᵃᶜᵃ[1] == Δ/2
    @test grid.zᵃᵃᶜ[1] == Δ/2

    return nothing
end

function test_regular_rectilinear_correct_end_faces(FT)
    N = 4
    L = 2.0
    Δ = L / N

    grid = RectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    @test grid.xᶠᵃᵃ[N+1] == L
    @test grid.yᵃᶠᵃ[N+2] == L + Δ
    @test grid.zᵃᵃᶠ[N+2] == L + Δ

    return nothing
end

function test_regular_rectilinear_ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    Hx, Hy, Hz = 1, 2, 1

    grid = RectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(Hx, Hy, Hz),
                                  topology=(Bounded, Bounded, Bounded))

    @test length(grid.xᶜᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶜᵃ) == Ny + 2Hy
    @test length(grid.zᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.xᶠᵃᵃ) == Nx + 1 + 2Hx
    @test length(grid.yᵃᶠᵃ) == Ny + 1 + 2Hy
    @test length(grid.zᵃᵃᶠ) == Nz + 1 + 2Hz

    return nothing
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function test_regular_rectilinear_no_roundoff_error_in_ranges(FT)
    Nx = Ny = 1
    Nz = 64
    Hz = 1

    grid = RectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, π/2), halo=(1, 1, Hz))

    @test length(grid.zᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.zᵃᵃᶠ) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_grid_properties_are_same_type(FT)
    grid = RectilinearGrid(FT, size=(10, 10, 10), extent=(1, 1//7, 2π))

    @test grid.Lx isa FT
    @test grid.Ly isa FT
    @test grid.Lz isa FT
    @test grid.Δx isa FT
    @test grid.Δy isa FT
    @test grid.Δz isa FT

    @test eltype(grid.xᶠᵃᵃ) == FT
    @test eltype(grid.yᵃᶠᵃ) == FT
    @test eltype(grid.zᵃᵃᶠ) == FT
    @test eltype(grid.xᶜᵃᵃ) == FT
    @test eltype(grid.yᵃᶜᵃ) == FT
    @test eltype(grid.zᵃᵃᶜ) == FT

    return nothing
end

function test_xnode_ynode_znode_are_correct(FT)
    N = 3
    grid = RectilinearGrid(FT, size=(N, N, N), x=(0, π), y=(0, π), z=(0, π),
                                  topology=(Periodic, Periodic, Bounded))

    @test xnode(Center(), 2, grid) ≈ FT(π/2)
    @test ynode(Center(), 2, grid) ≈ FT(π/2)
    @test znode(Center(), 2, grid) ≈ FT(π/2)

    @test xnode(Face(), 2, grid) ≈ FT(π/3)
    @test ynode(Face(), 2, grid) ≈ FT(π/3)
    @test znode(Face(), 2, grid) ≈ FT(π/3)

    return nothing
end

function test_regular_rectilinear_constructor_errors(FT)
    @test isbitstype(typeof(RectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, 1))))

    @test_throws ArgumentError RectilinearGrid(FT, size=(32,), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 64), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32, 16), extent=(1, 1, 1))

    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32.0), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(20.1, 32, 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, nothing, 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, "32", 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32), extent=(1, nothing, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32), extent=(1, "1", 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1.0, 1, 1))

    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), y=[1, 2])
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), z=(-π, π))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=1, y=2, z=3)
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(0, 2), z=4)
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=("0", "1"))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=(1, 2, 3))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(1, 0), y=(1//7, 5//7), z=(1, 2))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1))
    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))

    @test_throws ArgumentError RectilinearGrid(FT, size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Flux))

    @test_throws ArgumentError RectilinearGrid(FT, topology=(Flat, Periodic, Periodic), size=(16, 16, 16), extent=1)
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Periodic, Flat, Periodic), size=(16, 16, 16), extent=(1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=(16, 16, 16), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=(16, 16),     extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Periodic, Periodic, Flat), size=16,           extent=(1, 1, 1))

    @test_throws ArgumentError RectilinearGrid(FT, topology=(Periodic, Flat, Flat), size=16, extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Flat, Periodic, Flat), size=16, extent=(1, 1))
    @test_throws ArgumentError RectilinearGrid(FT, topology=(Flat, Flat, Periodic), size=(16, 16), extent=1)

    @test_throws ArgumentError RectilinearGrid(FT, topology=(Flat, Flat, Flat), size=16, extent=1)

    return nothing
end

function flat_size_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RectilinearGrid(FT; size=size, topology=topology, extent=extent)
    return grid.Nx, grid.Ny, grid.Nz
end

function flat_halo_regular_rectilinear_grid(FT; topology, size, halo, extent)
    grid = RectilinearGrid(FT; size=size, halo=halo, topology=topology, extent=extent)
    return grid.Hx, grid.Hy, grid.Hz
end

function flat_extent_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RectilinearGrid(FT; size=size, topology=topology, extent=extent)
    return grid.Lx, grid.Ly, grid.Lz
end

function test_flat_size_regular_rectilinear_grid(FT)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) === (1, 2, 3)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(2, 3), extent=(1, 1)) === (2, 1, 3)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(2, 3), extent=(1, 1)) === (2, 3, 1)

    @test flat_size_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) === (1, 2, 3)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(2, 3), extent=(1, 1)) === (2, 1, 3)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(2, 3), extent=(1, 1)) === (2, 3, 1)

    @test flat_size_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Flat), size=2, extent=1) === (2, 1, 1)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Flat), size=2, extent=1) === (1, 2, 1)
    @test flat_size_regular_rectilinear_grid(FT, topology=(Flat, Flat, Bounded),  size=2, extent=1) === (1, 1, 2)

    @test flat_size_regular_rectilinear_grid(FT, topology=(Flat, Flat, Flat), size=(), extent=()) === (1, 1, 1)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(1, 1), extent=(1, 1), halo=nothing) === (0, 1, 1)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(1, 1), extent=(1, 1), halo=nothing) === (1, 0, 1)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(1, 1), extent=(1, 1), halo=nothing) === (1, 1, 0)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(1, 1), extent=(1, 1), halo=(2, 3)) === (0, 2, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(1, 1), extent=(1, 1), halo=(2, 3)) === (2, 0, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(1, 1), extent=(1, 1), halo=(2, 3)) === (2, 3, 0)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Flat), size=1, extent=1, halo=2) === (2, 0, 0)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Flat), size=1, extent=1, halo=2) === (0, 2, 0)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Flat, Bounded),  size=1, extent=1, halo=2) === (0, 0, 2)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Flat, Flat), size=(), extent=(), halo=()) === (0, 0, 0)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) == (0, 1, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Periodic), size=(2, 3), extent=(1, 1)) == (1, 0, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Periodic, Flat), size=(2, 3), extent=(1, 1)) == (1, 1, 0)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Flat), size=2, extent=1) == (1, 0, 0)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Flat), size=2, extent=1) == (0, 1, 0)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Flat, Periodic), size=2, extent=1) == (0, 0, 1)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Flat, Flat), size=(), extent=()) == (0, 0, 0)

    return nothing
end

#####
##### Vertically stretched grids
#####

function test_vertically_stretched_grid_properties_are_same_type(FT, arch)
    grid = RectilinearGrid(FT, architecture=arch, size=(1, 1, 16), x=(0,1), y=(0,1), z=collect(0:16))

    @test grid.Lx isa FT
    @test grid.Ly isa FT
    @test grid.Lz isa FT
    @test grid.Δx isa FT
    @test grid.Δy isa FT

    @test eltype(grid.xᶠᵃᵃ) == FT
    @test eltype(grid.xᶜᵃᵃ) == FT
    @test eltype(grid.yᵃᶠᵃ) == FT
    @test eltype(grid.yᵃᶜᵃ) == FT
    @test eltype(grid.zᵃᵃᶠ) == FT
    @test eltype(grid.zᵃᵃᶜ) == FT

    @test eltype(grid.Δzᵃᵃᶜ) == FT
    @test eltype(grid.Δzᵃᵃᶠ) == FT

    return nothing
end

function test_architecturally_correct_stretched_grid(FT, arch, zᵃᵃᶠ)
    grid = RectilinearGrid(FT, architecture=arch, size=(1, 1, length(zᵃᵃᶠ)-1), x=(0, 1), y=(0, 1), z=zᵃᵃᶠ)

    ArrayType = array_type(arch)
    @test grid.zᵃᵃᶠ  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.zᵃᵃᶜ  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.Δzᵃᵃᶠ isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.Δzᵃᵃᶜ isa OffsetArray{FT, 1, <:ArrayType}

    return nothing
end

function test_correct_constant_grid_spacings(FT, Nz)
    grid = RectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=collect(0:Nz))

    @test all(grid.Δzᵃᵃᶜ .== 1)
    @test all(grid.Δzᵃᵃᶠ .== 1)

    return nothing
end

function test_correct_quadratic_grid_spacings(FT, Nz)
    grid = RectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=collect(0:Nz).^2)

     zᵃᵃᶠ(k) = (k-1)^2
     zᵃᵃᶜ(k) = (k^2 + (k-1)^2) / 2
    Δzᵃᵃᶠ(k) = k^2 - (k-1)^2
    Δzᵃᵃᶜ(k) = zᵃᵃᶜ(k+1) - zᵃᵃᶜ(k)

    @test all(isapprox.(  grid.zᵃᵃᶠ[1:Nz+1],  zᵃᵃᶠ.(1:Nz+1) ))
    @test all(isapprox.(  grid.zᵃᵃᶜ[1:Nz],    zᵃᵃᶜ.(1:Nz)   ))
    @test all(isapprox.( grid.Δzᵃᵃᶜ[1:Nz],   Δzᵃᵃᶠ.(1:Nz)   ))

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zᵃᵃᶠ
    @test all(isapprox.( grid.Δzᵃᵃᶠ[2:Nz-1], Δzᵃᵃᶜ.(2:Nz-1) ))

    return nothing
end

function test_correct_tanh_grid_spacings(FT, Nz)
    S = 3  # Stretching factor
    zᵃᵃᶠ(k) = tanh(S * (2 * (k - 1) / Nz - 1)) / tanh(S)

    grid = RectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=zᵃᵃᶠ)

     zᵃᵃᶜ(k) = (zᵃᵃᶠ(k) + zᵃᵃᶠ(k+1)) / 2
    Δzᵃᵃᶠ(k) = zᵃᵃᶠ(k+1) - zᵃᵃᶠ(k)
    Δzᵃᵃᶜ(k) = zᵃᵃᶜ(k+1) - zᵃᵃᶜ(k)

    @test all(isapprox.(  grid.zᵃᵃᶠ[1:Nz+1],  zᵃᵃᶠ.(1:Nz+1) ))
    @test all(isapprox.(  grid.zᵃᵃᶜ[1:Nz],    zᵃᵃᶜ.(1:Nz)   ))
    @test all(isapprox.( grid.Δzᵃᵃᶜ[1:Nz],   Δzᵃᵃᶠ.(1:Nz)   ))

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zᵃᵃᶠ
    @test all(isapprox.( grid.Δzᵃᵃᶠ[2:Nz-1], Δzᵃᵃᶜ.(2:Nz-1) ))

   return nothing
end

#####
##### Latitude-longitude grid tests
#####

function test_basic_lat_lon_bounded_domain(FT)
    Nλ = Nφ = 18
    Hλ = Hφ = 1

    grid = LatitudeLongitudeGrid(FT, size=(Nλ, Nφ, 1), longitude=(-90, 90), latitude=(-45, 45), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Bounded, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 180
    @test grid.Ly == 90
    @test grid.Lz == 1

    @test grid.Δλᶠᵃᵃ == 10
    @test grid.Δφᵃᶠᵃ == 5
    @test grid.Δzᵃᵃᶜ == 1
    @test grid.Δzᵃᵃᶠ == 1

    @test length(grid.λᶠᵃᵃ) == Nλ + 2Hλ + 1
    @test length(grid.λᶜᵃᵃ) == Nλ + 2Hλ

    @test length(grid.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid.φᵃᶜᵃ) == Nφ + 2Hφ

    @test grid.λᶠᵃᵃ[1] == -90
    @test grid.λᶠᵃᵃ[Nλ+1] == 90

    @test grid.φᵃᶠᵃ[1] == -45
    @test grid.φᵃᶠᵃ[Nφ+1] == 45

    @test grid.λᶠᵃᵃ[0] == -90 - grid.Δλᶠᵃᵃ
    @test grid.λᶠᵃᵃ[Nλ+2] == 90 + grid.Δλᶠᵃᵃ

    @test grid.φᵃᶠᵃ[0] == -45 - grid.Δφᵃᶠᵃ
    @test grid.φᵃᶠᵃ[Nφ+2] == 45 + grid.Δφᵃᶠᵃ

    @test all(diff(grid.λᶠᵃᵃ.parent) .== grid.Δλᶠᵃᵃ)
    @test all(diff(grid.λᶜᵃᵃ.parent) .== grid.Δλᶜᵃᵃ)

    @test all(diff(grid.φᵃᶠᵃ.parent) .== grid.Δφᵃᶠᵃ)
    @test all(diff(grid.φᵃᶜᵃ.parent) .== grid.Δφᵃᶜᵃ)

    return nothing
end

function test_basic_lat_lon_periodic_domain(FT)
    Nλ = 36
    Nφ = 32
    Hλ = Hφ = 1

    grid = LatitudeLongitudeGrid(FT, size=(Nλ, Nφ, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Periodic, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 360
    @test grid.Ly == 160
    @test grid.Lz == 1

    @test grid.Δλᶠᵃᵃ == 10
    @test grid.Δφᵃᶠᵃ == 5
    @test grid.Δzᵃᵃᶜ == 1
    @test grid.Δzᵃᵃᶠ == 1

    @test length(grid.λᶠᵃᵃ) == Nλ + 2Hλ
    @test length(grid.λᶜᵃᵃ) == Nλ + 2Hλ

    @test length(grid.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid.φᵃᶜᵃ) == Nφ + 2Hφ

    @test grid.λᶠᵃᵃ[1] == -180
    @test grid.λᶠᵃᵃ[Nλ] == 180 - grid.Δλᶠᵃᵃ

    @test grid.φᵃᶠᵃ[1] == -80
    @test grid.φᵃᶠᵃ[Nφ+1] == 80

    @test grid.λᶠᵃᵃ[0] == -180 - grid.Δλᶠᵃᵃ
    @test grid.λᶠᵃᵃ[Nλ+1] == 180

    @test grid.φᵃᶠᵃ[0] == -80 - grid.Δφᵃᶠᵃ
    @test grid.φᵃᶠᵃ[Nφ+2] == 80 + grid.Δφᵃᶠᵃ

    @test all(diff(grid.λᶠᵃᵃ.parent) .== grid.Δλᶠᵃᵃ)
    @test all(diff(grid.λᶜᵃᵃ.parent) .== grid.Δλᶜᵃᵃ)

    @test all(diff(grid.φᵃᶠᵃ.parent) .== grid.Δφᵃᶠᵃ)
    @test all(diff(grid.φᵃᶜᵃ.parent) .== grid.Δφᵃᶜᵃ)

    return nothing
end

function test_basic_lat_lon_general_grid(FT)

    (Nλ, Nφ, Nz) = size = (24, 16, 16)
    (Hλ, Hφ, Hz) = halo = ( 1,  1,  1)

    lat = (-80,   80)
    lon = (-180, 180) 
    zᵣ  = (-100,   0)

    Λ₁  = (lat[1], lon[1], zᵣ[1])
    Λₙ  = (lat[2], lon[2], zᵣ[2])

    (Lλ, Lφ, Lz) = L = @. Λₙ - Λ₁ 
    
    grid_reg = LatitudeLongitudeGrid(FT, size=size, halo=halo, latitude=lat, longitude=lon, z=zᵣ)

    @test typeof(grid_reg.Δzᵃᵃᶜ) == typeof(grid_reg.Δzᵃᵃᶠ) == FT

    Δz = grid_reg.Δzᵃᵃᶜ
    zₛ = -Lz:Δz:0

    grid_str = LatitudeLongitudeGrid(FT, size=size, halo=halo, latitude=lat, longitude=lon, z=zₛ)

    @test length(grid_str.λᶠᵃᵃ) == length(grid_reg.λᶠᵃᵃ) == Nλ + 2Hλ
    @test length(grid_str.λᶜᵃᵃ) == length(grid_reg.λᶜᵃᵃ) == Nλ + 2Hλ
        
    @test length(grid_str.φᵃᶠᵃ) == length(grid_reg.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid_str.φᵃᶜᵃ) == length(grid_reg.φᵃᶜᵃ) == Nφ + 2Hφ
    
    @test length(grid_str.zᵃᵃᶠ) == length(grid_reg.zᵃᵃᶠ) == Nz + 2Hz + 1
    @test length(grid_str.zᵃᵃᶜ) == length(grid_reg.zᵃᵃᶜ) == Nz + 2Hz
    
    @test length(grid_str.Δzᵃᵃᶠ) == Nz + 2Hz + 1
    @test length(grid_str.Δzᵃᵃᶜ) == Nz + 2Hz 

    @test all(grid_str.λᶜᵃᵃ == grid_reg.λᶜᵃᵃ) 
    @test all(grid_str.λᶠᵃᵃ == grid_reg.λᶠᵃᵃ)
    @test all(grid_str.φᵃᶜᵃ == grid_reg.φᵃᶜᵃ)
    @test all(grid_str.φᵃᶠᵃ == grid_reg.φᵃᶠᵃ)
    @test all(grid_str.zᵃᵃᶜ == grid_reg.zᵃᵃᶜ)
    @test all(grid_str.zᵃᵃᶠ == grid_reg.zᵃᵃᶠ)

    @test sum(grid_str.Δzᵃᵃᶜ) == grid_reg.Δzᵃᵃᶜ * length(grid_str.Δzᵃᵃᶜ)
    @test sum(grid_str.Δzᵃᵃᶠ) == grid_reg.Δzᵃᵃᶠ * length(grid_str.Δzᵃᵃᶠ)

    return nothing
end

#####
##### Conformal cubed sphere face grid
#####

function test_cubed_sphere_face_array_size(FT)
    grid = ConformalCubedSphereFaceGrid(FT, size=(10, 10, 1), z=(0, 1))

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz

    @test grid.λᶜᶜᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.λᶠᶜᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.λᶜᶠᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.λᶠᶠᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.φᶜᶜᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.φᶠᶜᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.φᶜᶠᵃ isa OffsetArray{FT, 2, <:Array}
    @test grid.φᶠᶠᵃ isa OffsetArray{FT, 2, <:Array}

    @test size(grid.λᶜᶜᵃ) == (Nx + 2Hx,     Ny + 2Hy    )
    @test size(grid.λᶠᶜᵃ) == (Nx + 2Hx + 1, Ny + 2Hy    )
    @test size(grid.λᶜᶠᵃ) == (Nx + 2Hx,     Ny + 2Hy + 1)
    @test size(grid.λᶠᶠᵃ) == (Nx + 2Hx + 1, Ny + 2Hy + 1)

    @test size(grid.φᶜᶜᵃ) == (Nx + 2Hx,     Ny + 2Hy    )
    @test size(grid.φᶠᶜᵃ) == (Nx + 2Hx + 1, Ny + 2Hy    )
    @test size(grid.φᶜᶠᵃ) == (Nx + 2Hx,     Ny + 2Hy + 1)
    @test size(grid.φᶠᶠᵃ) == (Nx + 2Hx + 1, Ny + 2Hy + 1)

    return nothing
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
    end

    @testset "Regular rectilinear grid" begin
        @info "  Testing regular rectilinear grid..."

        @testset "Grid initialization" begin
            @info "    Testing grid initialization..."

            for FT in float_types
                test_regular_rectilinear_correct_size(FT)
                test_regular_rectilinear_correct_extent(FT)
                test_regular_rectilinear_correct_coordinate_lengths(FT)
                test_regular_rectilinear_correct_halo_size(FT)
                test_regular_rectilinear_correct_halo_faces(FT)
                test_regular_rectilinear_correct_first_cells(FT)
                test_regular_rectilinear_correct_end_faces(FT)
                test_regular_rectilinear_ranges_have_correct_length(FT)
                test_regular_rectilinear_no_roundoff_error_in_ranges(FT)
                test_regular_rectilinear_grid_properties_are_same_type(FT)
                test_xnode_ynode_znode_are_correct(FT)
            end
        end

        @testset "Grid dimensions" begin
            @info "    Testing grid constructor errors..."

            for FT in float_types
                test_regular_rectilinear_constructor_errors(FT)
            end
        end

        @testset "Grids with flat dimensions" begin
            @info "    Testing construction of grids with Flat dimensions..."

            for FT in float_types
                test_flat_size_regular_rectilinear_grid(FT)
            end
        end

        # Testing show function
        topo = (Periodic, Periodic, Periodic)
        
        grid = RectilinearGrid(topology=topo, size=(3, 7, 9), x=(0, 1), y=(-π, π), z=(0, 2π))

        @test try
            CUDA.allowscalar(false)           
            show(grid); println()
            CUDA.allowscalar(true)
            true
        catch err
            println("error in show(::RectilinearGrid)")
            println(sprint(showerror, err))
            false
        end
        
        @test grid isa RectilinearGrid
    end

    @testset "Vertically stretched rectilinear grid" begin
        @info "  Testing vertically stretched rectilinear grid..."

        for arch in archs, FT in float_types
            @testset "Vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]" begin
                @info "    Testing vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]..."

                test_vertically_stretched_grid_properties_are_same_type(FT, arch)

                zᵃᵃᶠ1 = collect(0:10).^2
                zᵃᵃᶠ2 = [1, 3, 5, 10, 15, 33, 50]
                for zᵃᵃᶠ in [zᵃᵃᶠ1, zᵃᵃᶠ2]
                    test_architecturally_correct_stretched_grid(FT, arch, zᵃᵃᶠ)
                end
            end

            @testset "Vertically stretched rectilinear grid spacings [$(typeof(arch)), $FT]" begin
                @info "    Testing vertically stretched rectilinear grid spacings [$(typeof(arch)), $FT]..."
                for Nz in [16, 17]
                    test_correct_constant_grid_spacings(FT, Nz)
                    test_correct_quadratic_grid_spacings(FT, Nz)
                    test_correct_tanh_grid_spacings(FT, Nz)
                end
            end

            # Testing show function
            Nz = 20
            grid = RectilinearGrid(architecture=arch, size=(1, 1, Nz-1), x=(0, 1), y=(0, 1), z=collect(0:Nz).^2)
            
            @test try
            CUDA.allowscalar(false)           
            show(grid); println()
            CUDA.allowscalar(true)
                true
            catch err
                println("error in show(::RectilinearGrid)")
                println(sprint(showerror, err))
                false
            end
            
            @test grid isa RectilinearGrid
        end
    end

    @testset "Latitude-longitude grid" begin
        @info "  Testing general latitude-longitude grid..."

        for FT in float_types
            test_basic_lat_lon_bounded_domain(FT)
            test_basic_lat_lon_periodic_domain(FT)
            test_basic_lat_lon_general_grid(FT)
        end

        # Testing show function for regular grid
        grid = LatitudeLongitudeGrid(size=(36, 32, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1))
    
        @test try
            CUDA.allowscalar(false)           
            show(grid); println()
            CUDA.allowscalar(true)
            true
        catch err
            println("error in show(::LatitudeLongitudeGrid)")
            println(sprint(showerror, err))
            false
        end

        @test grid isa LatitudeLongitudeGrid

        # Testing show function for stretched grid
        grid = LatitudeLongitudeGrid(size=(36, 32, 10), longitude=(-180, 180), latitude=(-80, 80), z=collect(0:10))

        @test try
            CUDA.allowscalar(false)           
            show(grid); println()
            CUDA.allowscalar(true)
            true
        catch err
            println("error in show(::LatitudeLongitudeGrid)")
            println(sprint(showerror, err))
            false
        end

        @test grid isa LatitudeLongitudeGrid
    end

    @testset "Conformal cubed sphere face grid" begin
        @info "  Testing conformal cubed sphere face grid..."

        for FT in float_types
            test_cubed_sphere_face_array_size(Float64)
        end

        # Testing show function
        grid = ConformalCubedSphereFaceGrid(size=(10, 10, 1), z=(0, 1))
    
        @test try
            CUDA.allowscalar(false)           
            show(grid); println()
            CUDA.allowscalar(true)
            true
        catch err
            println("error in show(::ConformalCubedSphereFaceGrid)")
            println(sprint(showerror, err))
            false
        end

        @test grid isa ConformalCubedSphereFaceGrid
    end

    @testset "Conformal cubed sphere face grid from file" begin
        @info "  Testing conformal cubed sphere face grid construction from file..."

        cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"

        for face in 1:6
            grid = ConformalCubedSphereFaceGrid(cs32_filepath, face=face, Nz=1, z=(-1, 0))
            @test grid isa ConformalCubedSphereFaceGrid
        end
    end
end
