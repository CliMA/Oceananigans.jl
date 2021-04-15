using DataDeps

using Oceananigans.Grids: total_extent, halo_size

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

#####
##### Regular rectilinear grids
#####

function test_regular_rectilinear_correct_size(FT)
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π))

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
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), x=(1, 2), y=(π, 3π), z=(0, 4))

    @test grid.Lx ≈ 1
    @test grid.Ly ≈ 2π
    @test grid.Lz ≈ 4

    return nothing
end

function test_regular_rectilinear_correct_coordinate_lengths(FT)
    grid = RegularRectilinearGrid(FT, size=(2, 3, 4), extent=(1, 1, 1), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    @test length(grid.xC) == Nx + 2Hx
    @test length(grid.yC) == Ny + 2Hy
    @test length(grid.zC) == Nz + 2Hz
    @test length(grid.xF) == Nx + 2Hx
    @test length(grid.yF) == Ny + 2Hy + 1
    @test length(grid.zF) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_correct_halo_size(FT)
    grid = RegularRectilinearGrid(FT, size=(4, 6, 8), extent=(2π, 4π, 9π), halo=(1, 2, 3))

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
    grid = RegularRectilinearGrid(FT, topology=topo, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xF[0] == - H * Δ
    @test grid.yF[0] == - H * Δ
    @test grid.zF[0] == - H * Δ

    @test grid.xF[N+1] == L  # Periodic
    @test grid.yF[N+2] == L + H * Δ
    @test grid.zF[N+2] == L + H * Δ

    return nothing
end

function test_regular_rectilinear_correct_first_cells(FT)
    N = 4
    H = 1
    L = 4.0
    Δ = L / N

    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xC[1] == Δ/2
    @test grid.yC[1] == Δ/2
    @test grid.zC[1] == Δ/2

    return nothing
end

function test_regular_rectilinear_correct_end_faces(FT)
    N = 4
    L = 2.0
    Δ = L / N

    grid = RegularRectilinearGrid(FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    @test grid.xF[N+1] == L
    @test grid.yF[N+2] == L + Δ
    @test grid.zF[N+2] == L + Δ

    return nothing
end

function test_regular_rectilinear_ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    Hx, Hy, Hz = 1, 2, 1

    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(Hx, Hy, Hz),
                                  topology=(Bounded, Bounded, Bounded))

    @test length(grid.xC) == Nx + 2Hx
    @test length(grid.yC) == Ny + 2Hy
    @test length(grid.zC) == Nz + 2Hz
    @test length(grid.xF) == Nx + 1 + 2Hx
    @test length(grid.yF) == Ny + 1 + 2Hy
    @test length(grid.zF) == Nz + 1 + 2Hz

    return nothing
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function test_regular_rectilinear_no_roundoff_error_in_ranges(FT)
    Nx = Ny = 1
    Nz = 64
    Hz = 1

    grid = RegularRectilinearGrid(FT, size=(Nx, Ny, Nz), extent=(1, 1, π/2), halo=(1, 1, Hz))

    @test length(grid.zC) == Nz + 2Hz
    @test length(grid.zF) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_grid_properties_are_same_type(FT)
    grid = RegularRectilinearGrid(FT, size=(10, 10, 10), extent=(1, 1//7, 2π))

    @test grid.Lx isa FT
    @test grid.Ly isa FT
    @test grid.Lz isa FT
    @test grid.Δx isa FT
    @test grid.Δy isa FT
    @test grid.Δz isa FT

    @test eltype(grid.xF) == FT
    @test eltype(grid.yF) == FT
    @test eltype(grid.zF) == FT
    @test eltype(grid.xC) == FT
    @test eltype(grid.yC) == FT
    @test eltype(grid.zC) == FT

    return nothing
end

function test_xnode_ynode_znode_are_correct(FT)
    N = 3
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

function test_regular_rectilinear_constructor_errors(FT)
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
    grid = VerticallyStretchedRectilinearGrid(FT, architecture=arch, size=(1, 1, 16), x=(0,1), y=(0,1), zF=collect(0:16))

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

function test_architecturally_correct_stretched_grid(FT, arch, zF)
    grid = VerticallyStretchedRectilinearGrid(FT, architecture=arch, size=(1, 1, length(zF)-1), x=(0, 1), y=(0, 1), zF=zF)

    ArrayType = array_type(arch)
    @test grid.zᵃᵃᶠ  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.zᵃᵃᶜ  isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.Δzᵃᵃᶠ isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.Δzᵃᵃᶜ isa OffsetArray{FT, 1, <:ArrayType}

    return nothing
end

function test_correct_constant_grid_spacings(FT, Nz)
    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz))

    @test all(grid.Δzᵃᵃᶜ .== 1)
    @test all(grid.Δzᵃᵃᶠ .== 1)

    return nothing
end

function test_correct_quadratic_grid_spacings(FT, Nz)
    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz).^2)

     zF(k) = (k-1)^2
     zC(k) = (k^2 + (k-1)^2) / 2
    ΔzF(k) = k^2 - (k-1)^2
    ΔzC(k) = zC(k+1) - zC(k)

    @test all(isapprox.(  grid.zᵃᵃᶠ[1:Nz+1],  zF.(1:Nz+1) ))
    @test all(isapprox.(  grid.zᵃᵃᶜ[1:Nz],    zC.(1:Nz)   ))
    @test all(isapprox.( grid.Δzᵃᵃᶜ[1:Nz],   ΔzF.(1:Nz)   ))

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zF
    @test all(isapprox.( grid.Δzᵃᵃᶠ[2:Nz-1], ΔzC.(2:Nz-1) ))

    return nothing
end

function test_correct_tanh_grid_spacings(FT, Nz)
    S = 3  # Stretching factor
    zF(k) = tanh(S * (2 * (k - 1) / Nz - 1)) / tanh(S)

    grid = VerticallyStretchedRectilinearGrid(FT, size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=zF)

     zC(k) = (zF(k) + zF(k+1)) / 2
    ΔzF(k) = zF(k+1) - zF(k)
    ΔzC(k) = zC(k+1) - zC(k)

    @test all(isapprox.(  grid.zᵃᵃᶠ[1:Nz+1],  zF.(1:Nz+1) ))
    @test all(isapprox.(  grid.zᵃᵃᶜ[1:Nz],    zC.(1:Nz)   ))
    @test all(isapprox.( grid.Δzᵃᵃᶜ[1:Nz],   ΔzF.(1:Nz)   ))

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zF
    @test all(isapprox.( grid.Δzᵃᵃᶠ[2:Nz-1], ΔzC.(2:Nz-1) ))

   return nothing
end

#####
##### Regular latitude-longitude grid tests
#####

function test_basic_lat_lon_bounded_domain(FT)
    Nλ = Nφ = 18
    Hλ = Hφ = 1

    grid = RegularLatitudeLongitudeGrid(FT, size=(Nλ, Nφ, 1), longitude=(-90, 90), latitude=(-45, 45), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Bounded, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 180
    @test grid.Ly == 90
    @test grid.Lz == 1

    @test grid.Δλ == 10
    @test grid.Δφ == 5
    @test grid.Δz == 1

    @test length(grid.λᶠᵃᵃ) == Nλ + 2Hλ + 1
    @test length(grid.λᶜᵃᵃ) == Nλ + 2Hλ

    @test length(grid.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid.φᵃᶜᵃ) == Nφ + 2Hφ

    @test grid.λᶠᵃᵃ[1] == -90
    @test grid.λᶠᵃᵃ[Nλ+1] == 90

    @test grid.φᵃᶠᵃ[1] == -45
    @test grid.φᵃᶠᵃ[Nφ+1] == 45

    @test grid.λᶠᵃᵃ[0] == -90 - grid.Δλ
    @test grid.λᶠᵃᵃ[Nλ+2] == 90 + grid.Δλ

    @test grid.φᵃᶠᵃ[0] == -45 - grid.Δφ
    @test grid.φᵃᶠᵃ[Nφ+2] == 45 + grid.Δφ

    @test all(diff(grid.λᶠᵃᵃ.parent) .== grid.Δλ)
    @test all(diff(grid.λᶜᵃᵃ.parent) .== grid.Δλ)

    @test all(diff(grid.φᵃᶠᵃ.parent) .== grid.Δφ)
    @test all(diff(grid.φᵃᶜᵃ.parent) .== grid.Δφ)

    return nothing
end

function test_basic_lat_lon_periodic_domain(FT)
    Nλ = 36
    Nφ = 32
    Hλ = Hφ = 1

    grid = RegularLatitudeLongitudeGrid(FT, size=(Nλ, Nφ, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Periodic, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 360
    @test grid.Ly == 160
    @test grid.Lz == 1

    @test grid.Δλ == 10
    @test grid.Δφ == 5
    @test grid.Δz == 1

    @test length(grid.λᶠᵃᵃ) == Nλ + 2Hλ
    @test length(grid.λᶜᵃᵃ) == Nλ + 2Hλ

    @test length(grid.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid.φᵃᶜᵃ) == Nφ + 2Hφ

    @test grid.λᶠᵃᵃ[1] == -180
    @test grid.λᶠᵃᵃ[Nλ] == 180 - grid.Δλ

    @test grid.φᵃᶠᵃ[1] == -80
    @test grid.φᵃᶠᵃ[Nφ+1] == 80

    @test grid.λᶠᵃᵃ[0] == -180 - grid.Δλ
    @test grid.λᶠᵃᵃ[Nλ+1] == 180

    @test grid.φᵃᶠᵃ[0] == -80 - grid.Δφ
    @test grid.φᵃᶠᵃ[Nφ+2] == 80 + grid.Δφ

    @test all(diff(grid.λᶠᵃᵃ.parent) .== grid.Δλ)
    @test all(diff(grid.λᶜᵃᵃ.parent) .== grid.Δλ)

    @test all(diff(grid.φᵃᶠᵃ.parent) .== grid.Δφ)
    @test all(diff(grid.φᵃᶜᵃ.parent) .== grid.Δφ)

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
        grid = RegularRectilinearGrid(topology=topo, size=(3, 7, 9), x=(0, 1), y=(-π, π), z=(0, 2π))
        show(grid); println();
        @test grid isa RegularRectilinearGrid
    end

    @testset "Vertically stretched rectilinear grid" begin
        @info "  Testing vertically stretched rectilinear grid..."

        for arch in archs, FT in float_types
            @testset "Vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]" begin
                @info "    Testing vertically stretched rectilinear grid construction [$(typeof(arch)), $FT]..."

                test_vertically_stretched_grid_properties_are_same_type(FT, arch)

                zF1 = collect(0:10).^2
                zF2 = [1, 3, 5, 10, 15, 33, 50]
                for zF in [zF1, zF2]
                    test_architecturally_correct_stretched_grid(FT, arch, zF)
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
            grid = VerticallyStretchedRectilinearGrid(size=(1, 1, Nz), x=(0, 1), y=(0, 1), zF=collect(0:Nz).^2)
            show(grid); println();
            @test grid isa VerticallyStretchedRectilinearGrid
        end
    end

    @testset "Regular latitude-longitude grid" begin
        @info "  Testing regular latitude-longitude grid..."

        for FT in float_types
            test_basic_lat_lon_bounded_domain(FT)
            test_basic_lat_lon_periodic_domain(FT)
        end

        # Testing show function
        grid = RegularLatitudeLongitudeGrid(size=(36, 32, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1))
        show(grid); println();
        @test grid isa RegularLatitudeLongitudeGrid
    end

    @testset "Conformal cubed sphere face grid" begin
        @info "  Testing conformal cubed sphere face grid..."

        for FT in float_types
            test_cubed_sphere_face_array_size(Float64)
        end

        # Testing show function
        grid = ConformalCubedSphereFaceGrid(size=(10, 10, 1), z=(0, 1))
        show(grid); println();
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
