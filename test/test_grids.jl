include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: total_extent, ColumnEnsembleSize,
                          xspacings, yspacings, zspacings,
                          xnode, ynode, znode, λnode, φnode,
                          λspacings, φspacings

using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, ConformalCubedSpherePanelGrid

using Oceananigans.Operators: Δx, Δy, Δz, Δλ, Δφ, Ax, Ay, Az, volume
using Oceananigans.Operators: Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ

#####
##### Regular rectilinear grids
#####

function test_regular_rectilinear_correct_size(FT)
    grid = RectilinearGrid(CPU(), FT, size=(4, 6, 8), extent=(2π, 4π, 9π))

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
    grid = RectilinearGrid(CPU(), FT, size=(4, 6, 8), x=(1, 2), y=(π, 3π), z=(0, 4))

    @test grid.Lx ≈ 1
    @test grid.Ly ≈ 2π
    @test grid.Lz ≈ 4

    return nothing
end

function test_regular_rectilinear_correct_coordinate_lengths(FT)
    grid = RectilinearGrid(CPU(), FT, size=(2, 3, 4), extent=(1, 1, 1), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    Nx, Ny, Nz = size(grid)
    Hx, Hy, Hz = halo_size(grid)

    @test length(grid.xᶜᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶜᵃ) == Ny + 2Hy
    @test length(grid.xᶠᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶠᵃ) == Ny + 2Hy + 1

    @test length(grid.z.cᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.z.cᵃᵃᶠ) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_correct_halo_size(FT)
    grid = RectilinearGrid(CPU(), FT, size=(4, 6, 8), extent=(2π, 4π, 9π), halo=(1, 2, 3))

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
    grid = RectilinearGrid(CPU(), FT, topology=topo, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xᶠᵃᵃ[0] == - H * Δ
    @test grid.yᵃᶠᵃ[0] == - H * Δ
    @test grid.z.cᵃᵃᶠ[0] == - H * Δ

    @test grid.xᶠᵃᵃ[N+1] == L  # Periodic
    @test grid.yᵃᶠᵃ[N+2] == L + H * Δ
    @test grid.z.cᵃᵃᶠ[N+2] == L + H * Δ

    return nothing
end

function test_regular_rectilinear_correct_first_cells(FT)
    N = 4
    H = 1
    L = 4.0
    Δ = L / N

    grid = RectilinearGrid(CPU(), FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(H, H, H))

    @test grid.xᶜᵃᵃ[1] == Δ/2
    @test grid.yᵃᶜᵃ[1] == Δ/2
    @test grid.z.cᵃᵃᶜ[1] == Δ/2

    return nothing
end

function test_regular_rectilinear_correct_end_faces(FT)
    N = 4
    L = 2.0
    Δ = L / N

    grid = RectilinearGrid(CPU(), FT, size=(N, N, N), x=(0, L), y=(0, L), z=(0, L), halo=(1, 1, 1),
                                  topology=(Periodic, Bounded, Bounded))

    @test grid.xᶠᵃᵃ[N+1] == L
    @test grid.yᵃᶠᵃ[N+2] == L + Δ
    @test grid.z.cᵃᵃᶠ[N+2] == L + Δ

    return nothing
end

function test_regular_rectilinear_ranges_have_correct_length(FT)
    Nx, Ny, Nz = 8, 9, 10
    Hx, Hy, Hz = 1, 2, 1

    grid = RectilinearGrid(CPU(), FT, size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(Hx, Hy, Hz),
                                  topology=(Bounded, Bounded, Bounded))

    @test length(grid.xᶜᵃᵃ) == Nx + 2Hx
    @test length(grid.yᵃᶜᵃ) == Ny + 2Hy
    @test length(grid.xᶠᵃᵃ) == Nx + 1 + 2Hx
    @test length(grid.yᵃᶠᵃ) == Ny + 1 + 2Hy

    @test length(grid.z.cᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.z.cᵃᵃᶠ) == Nz + 1 + 2Hz

    return nothing
end

# See: https://github.com/climate-machine/Oceananigans.jl/issues/480
function test_regular_rectilinear_no_roundoff_error_in_ranges(FT)
    Nx = Ny = 1
    Nz = 64
    Hz = 1

    grid = RectilinearGrid(CPU(), FT, size=(Nx, Ny, Nz), extent=(1, 1, π/2), halo=(1, 1, Hz))

    @test length(grid.z.cᵃᵃᶜ) == Nz + 2Hz
    @test length(grid.z.cᵃᵃᶠ) == Nz + 2Hz + 1

    return nothing
end

function test_regular_rectilinear_grid_properties_are_same_type(FT)
    # Do this test two ways, one with defaults and one with explicit FT
    for method in (:explicit, :with_default)

        if method === :explicit
            grid = RectilinearGrid(CPU(), FT, size=(10, 10, 10), extent=(1, 1//7, 2π))
        elseif method === :with_default
            FT₀ = Oceananigans.defaults.FloatType
            Oceananigans.defaults.FloatType = FT
            grid = RectilinearGrid(CPU(), size=(10, 10, 10), extent=(1, 1//7, 2π))
            Oceananigans.defaults.FloatType = FT₀
        end

        @test grid.Lx isa FT
        @test grid.Ly isa FT
        @test grid.Lz isa FT
        @test grid.Δxᶠᵃᵃ isa FT
        @test grid.Δyᵃᶠᵃ isa FT
        @test grid.z.Δᵃᵃᶠ isa FT

        @test eltype(grid.xᶠᵃᵃ) == FT
        @test eltype(grid.yᵃᶠᵃ) == FT
        @test eltype(grid.z.cᵃᵃᶠ) == FT
        @test eltype(grid.xᶜᵃᵃ) == FT
        @test eltype(grid.yᵃᶜᵃ) == FT
        @test eltype(grid.z.cᵃᵃᶜ) == FT
    end

    return nothing
end

function test_regular_rectilinear_xnode_ynode_znode_and_spacings(arch, FT)

    @info "    Testing with ($FT) on ($arch)..."

    N = 3

    size=(N, N, N)
    topology = (Periodic, Periodic, Bounded)

    regular_spaced_grid = RectilinearGrid(arch, FT; size, topology,
                                          x=(0, π), y=(0, π), z=(0, π))

    domain = collect(range(0, stop=π, length=N+1))

    variably_spaced_grid = RectilinearGrid(arch, FT; size, topology,
                                           x=domain, y=domain, z=domain)

    ibg_regular_spaced_grid = ImmersedBoundaryGrid(regular_spaced_grid, GridFittedBottom((x, y) -> 0))

    ibg_variably_spaced_grid = ImmersedBoundaryGrid(variably_spaced_grid, GridFittedBottom((x, y) -> 0))

    grids_types = ["regularly spaced", "variably spaced", "IBG regularly spaced", "IBG variably spaced"]
    grids       = [regular_spaced_grid, variably_spaced_grid, ibg_regular_spaced_grid, ibg_variably_spaced_grid]

    for (grid_type, grid) in zip(grids_types, grids)
        @info "        Testing grid utils on $grid_type grid...."

        @test xnode(2, grid, Center()) ≈ FT(π/2)
        @test ynode(2, grid, Center()) ≈ FT(π/2)
        @test znode(2, grid, Center()) ≈ FT(π/2)

        @test xnode(2, grid, Face()) ≈ FT(π/3)
        @test ynode(2, grid, Face()) ≈ FT(π/3)
        @test znode(2, grid, Face()) ≈ FT(π/3)

        @test minimum_xspacing(grid) ≈ FT(π/3)
        @test minimum_yspacing(grid) ≈ FT(π/3)
        @test minimum_zspacing(grid) ≈ FT(π/3)

        @test all(xspacings(grid, Center()) .≈ FT(π/N))
        @test all(yspacings(grid, Center()) .≈ FT(π/N))
        @test all(zspacings(grid, Center()) .≈ FT(π/N))

        @test all(x ≈ FT(π/N) for x in xspacings(grid, Face()))
        @test all(y ≈ FT(π/N) for y in yspacings(grid, Face()))
        @test all(z ≈ FT(π/N) for z in zspacings(grid, Face()))

        @test all(xspacings(grid, Face()) .== xspacings(grid, Face(), Center(), Center()))
        @test all(yspacings(grid, Face()) .== yspacings(grid, Center(), Face(), Center()))
        @test all(zspacings(grid, Face()) .== zspacings(grid, Center(), Center(), Face()))

        @test Δx(1, 1, 1, grid, Face(), Center(), Center()) ≈ FT(π/N)
        @test Δy(1, 1, 1, grid, Center(), Face(), Center()) ≈ FT(π/N)
        @test Δz(1, 1, 1, grid, Center(), Center(), Face()) ≈ FT(π/N)
    end

    return nothing
end

function test_regular_rectilinear_constructor_errors(FT)
    @test isbitstype(typeof(RectilinearGrid(CPU(), FT, size=(16, 16, 16), extent=(1, 1, 1))))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32,), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 64), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32, 16), extent=(1, 1, 1))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32.0), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(20.1, 32, 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, nothing, 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, "32", 32), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32), extent=(1, nothing, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32), extent=(1, "1", 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(32, 32, 32), extent=(1, 1, 1), halo=(1.0, 1, 1))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), y=[1, 2])
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), z=(-π, π))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=1, y=2, z=3)
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(0, 1), y=(0, 2), z=4)
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=("0", "1"))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(-1//2, 1), y=(1//7, 5//7), z=(1, 2, 3))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(1, 0), y=(1//7, 5//7), z=(1, 2))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), x=(0, 1), y=(1, 5), z=(π, -π))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), extent=(1, 2, 3), x=(0, 1), y=(1, 5), z=(-π, π))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(16, 16, 16), extent=(1, 1, 1), topology=(Periodic, Periodic, Flux))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Flat, Periodic, Periodic), size=(16, 16, 16), extent=1)
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Periodic, Flat, Periodic), size=(16, 16, 16), extent=(1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Periodic, Periodic, Flat), size=(16, 16, 16), extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Periodic, Periodic, Flat), size=(16, 16),     extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Periodic, Periodic, Flat), size=16,           extent=(1, 1, 1))

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Periodic, Flat, Flat), size=16, extent=(1, 1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Flat, Periodic, Flat), size=16, extent=(1, 1))
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Flat, Flat, Periodic), size=(16, 16), extent=1)

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, topology=(Flat, Flat, Flat), size=16, extent=1)

    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(4, 4, 4), x=(0, 1), y=(0, 1), z=[-50.0, -30.0, -20.0, 0.0]) # too few z-faces
    @test_throws ArgumentError RectilinearGrid(CPU(), FT, size=(4, 4, 4), x=(0, 1), y=(0, 1), z=[-2000.0, -1000.0, -50.0, -30.0, -20.0, 0.0]) # too many z-faces

    return nothing
end

function flat_size_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RectilinearGrid(CPU(), FT; size, topology, extent)
    return grid.Nx, grid.Ny, grid.Nz
end

function flat_halo_regular_rectilinear_grid(FT; topology, size, halo, extent)
    grid = RectilinearGrid(CPU(), FT; size, halo, topology, extent)
    return grid.Hx, grid.Hy, grid.Hz
end

function flat_extent_regular_rectilinear_grid(FT; topology, size, extent)
    grid = RectilinearGrid(CPU(), FT; size, topology, extent)
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

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(3, 3), extent=(1, 1), halo=nothing) === (0, 3, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(3, 3), extent=(1, 1), halo=nothing) === (3, 0, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(3, 3), extent=(1, 1), halo=nothing) === (3, 3, 0)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(3, 3), extent=(1, 1), halo=(2, 3)) === (0, 2, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Bounded),  size=(3, 3), extent=(1, 1), halo=(2, 3)) === (2, 0, 3)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Bounded, Flat),  size=(3, 3), extent=(1, 1), halo=(2, 3)) === (2, 3, 0)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Flat), size=2, extent=1, halo=2) === (2, 0, 0)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Flat), size=2, extent=1, halo=2) === (0, 2, 0)
    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Flat, Bounded),  size=2, extent=1, halo=2) === (0, 0, 2)

    @test flat_halo_regular_rectilinear_grid(FT, topology=(Flat, Flat, Flat), size=(), extent=(), halo=()) === (0, 0, 0)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Periodic), size=(2, 3), extent=(1, 1)) == (1, 1, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Periodic), size=(2, 3), extent=(1, 1)) == (1, 1, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Periodic, Flat), size=(2, 3), extent=(1, 1)) == (1, 1, 1)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Periodic, Flat, Flat), size=2, extent=1) == (1, 1, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Periodic, Flat), size=2, extent=1) == (1, 1, 1)
    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Flat, Periodic), size=2, extent=1) == (1, 1, 1)

    @test flat_extent_regular_rectilinear_grid(FT, topology=(Flat, Flat, Flat), size=(), extent=()) == (1, 1, 1)

    return nothing
end

function test_grid_equality(arch)
    topo = (Periodic, Periodic, Bounded)
    Nx, Ny, Nz = 4, 7, 9
    grid1 = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), x=(0, 1), y=(-1, 1), z=(0, Nz))
    grid2 = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), x=(0, 1), y=(-1, 1), z=0:Nz)
    grid3 = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), x=(0, 1), y=(-1, 1), z=0:Nz)

    return grid1 == grid1 && grid2 == grid3 && grid1 !== grid3
end

function test_grid_equality_over_architectures()
    grid_cpu = RectilinearGrid(CPU(), topology=(Periodic, Periodic, Bounded), size=(3, 7, 9), x=(0, 1), y=(-1, 1), z=0:9)
    grid_gpu = RectilinearGrid(GPU(), topology=(Periodic, Periodic, Bounded), size=(3, 7, 9), x=(0, 1), y=(-1, 1), z=0:9)
    return grid_cpu == grid_gpu
end

function test_immersed_boundary_grid_equality(arch)
    underlying_grid = RectilinearGrid(arch, size=(4, 4, 4), extent=(1, 1, 1))

    ib1 = GridFittedBottom(-1/2)
    ibg1 = ImmersedBoundaryGrid(underlying_grid, ib1)
    @test ibg1 != underlying_grid
    @test ibg1.underlying_grid != ibg1
    @test ibg1 == ibg1

    ibg2 = ImmersedBoundaryGrid(underlying_grid, ib1)
    @test ibg1 == ibg2

    ib2 = PartialCellBottom(-1/2)
    ibg3 = ImmersedBoundaryGrid(underlying_grid, ib2)
    ibg4 = ImmersedBoundaryGrid(underlying_grid, ib2)
    ibg5 = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-1/2, minimum_fractional_cell_height=0.5))
    @test ibg3 == ibg4
    @test ibg3 != ibg1
    @test ibg3 != ibg5

    mask1 = zeros(Bool, 4, 4, 4)
    mask1[2:3, 2:3, 2:3] .= true
    ib3 = GridFittedBoundary(mask1)
    ibg6 = ImmersedBoundaryGrid(underlying_grid, ib3)
    ibg7 = ImmersedBoundaryGrid(underlying_grid, ib3)

    mask2 = zeros(Bool, 4, 4, 4)
    ib4 = GridFittedBoundary(mask2)
    ibg8 = ImmersedBoundaryGrid(underlying_grid, ib4)
    @test ibg6 != ibg8
    @test ibg7 != ibg8

    @test ibg1 != ibg3
    @test ibg1 != ibg6

    return true
end

#####
##### Vertically stretched grids
#####

function test_vertically_stretched_grid_properties_are_same_type(FT, arch)
    grid = RectilinearGrid(arch, FT, size=(1, 1, 16), x=(0,1), y=(0,1), z=collect(0:16))

    @test grid.Lx isa FT
    @test grid.Ly isa FT
    @test grid.Lz isa FT
    @test grid.Δxᶠᵃᵃ isa FT
    @test grid.Δyᵃᶠᵃ isa FT

    @test eltype(grid.xᶠᵃᵃ) == FT
    @test eltype(grid.xᶜᵃᵃ) == FT
    @test eltype(grid.yᵃᶠᵃ) == FT
    @test eltype(grid.yᵃᶜᵃ) == FT
    @test eltype(grid.z.cᵃᵃᶠ) == FT
    @test eltype(grid.z.cᵃᵃᶜ) == FT

    @test eltype(grid.z.Δᵃᵃᶜ) == FT
    @test eltype(grid.z.Δᵃᵃᶠ) == FT

    return nothing
end

function test_architecturally_correct_stretched_grid(FT, arch, zᵃᵃᶠ)
    grid = RectilinearGrid(arch, FT, size=(1, 1, length(zᵃᵃᶠ)-1), x=(0, 1), y=(0, 1), z=zᵃᵃᶠ)

    ArrayType = array_type(arch)
    @test grid.z.cᵃᵃᶠ isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.z.cᵃᵃᶜ isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.z.Δᵃᵃᶠ isa OffsetArray{FT, 1, <:ArrayType}
    @test grid.z.Δᵃᵃᶜ isa OffsetArray{FT, 1, <:ArrayType}

    return nothing
end

function test_rectilinear_grid_correct_spacings(FT, N)
    S = 3
    zᵃᵃᶠ(k) = tanh(S * (2 * (k - 1) / N - 1)) / tanh(S)

    # a grid with regular x-spacing, quadratic y-spacing, and tanh-like z-spacing
    grid = RectilinearGrid(CPU(), FT, size=(N, N, N), x=collect(0:N), y=collect(0:N).^2, z=zᵃᵃᶠ)

    @test all(grid.Δxᶜᵃᵃ .== 1)
    @test all(grid.Δxᶠᵃᵃ .== 1)

     yᵃᶠᵃ(j) = (j-1)^2
     yᵃᶜᵃ(j) = (j^2 + (j-1)^2) / 2
    Δyᵃᶠᵃ(j) = yᵃᶜᵃ(j) - yᵃᶜᵃ(j-1)
    Δyᵃᶜᵃ(j) = yᵃᶠᵃ(j+1) - yᵃᶠᵃ(j)

    @test all(isapprox.(  grid.yᵃᶠᵃ[1:N+1],  yᵃᶠᵃ.(1:N+1) ))
    @test all(isapprox.(  grid.yᵃᶜᵃ[1:N],    yᵃᶜᵃ.(1:N)   ))
    @test all(isapprox.( grid.Δyᵃᶜᵃ[1:N],   Δyᵃᶜᵃ.(1:N)   ))

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zᵃᵃᶠ
    @test all(isapprox.( grid.Δyᵃᶠᵃ[2:N], Δyᵃᶠᵃ.(2:N) ))

     zᵃᵃᶜ(k) = (zᵃᵃᶠ(k)   + zᵃᵃᶠ(k+1)) / 2
    Δzᵃᵃᶜ(k) =  zᵃᵃᶠ(k+1) - zᵃᵃᶠ(k)
    Δzᵃᵃᶠ(k) =  zᵃᵃᶜ(k)   - zᵃᵃᶜ(k-1)

    @test all(isapprox.(grid.z.cᵃᵃᶠ[1:N+1],  zᵃᵃᶠ.(1:N+1) ))
    @test all(isapprox.(grid.z.cᵃᵃᶜ[1:N],    zᵃᵃᶜ.(1:N)   ))
    @test all(isapprox.(grid.z.Δᵃᵃᶜ[1:N],   Δzᵃᵃᶜ.(1:N)   ))

    @test all(isapprox.(zspacings(grid, Face()),   reshape(grid.z.Δᵃᵃᶠ[1:N+1], 1, 1, N+1)))
    @test all(isapprox.(zspacings(grid, Center()), reshape(grid.z.Δᵃᵃᶜ[1:N], 1, 1, N)))

    @test Δz(1, 1, 2, grid, Center(), Center(), Face()) == grid.z.Δᵃᵃᶠ[2]

    @test minimum_zspacing(grid, Center(), Center(), Center()) ≈ minimum(grid.z.Δᵃᵃᶜ[1:grid.Nz])

    # Note that Δzᵃᵃᶠ[1] involves a halo point, which is not directly determined by
    # the user-supplied zᵃᵃᶠ
    @test all(isapprox.( grid.z.Δᵃᵃᶠ[2:N], Δzᵃᵃᶠ.(2:N) ))

    return nothing
end

#####
##### Latitude-longitude grid tests
#####

function test_basic_lat_lon_bounded_domain(FT)
    Nλ = Nφ = 18
    Hλ = Hφ = 1

    grid = LatitudeLongitudeGrid(CPU(), FT, size=(Nλ, Nφ, 1), longitude=(-90, 90), latitude=(-45, 45), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Bounded, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 180
    @test grid.Ly == 90
    @test grid.Lz == 1

    @test grid.Δλᶠᵃᵃ == 10
    @test grid.Δφᵃᶠᵃ == 5
    @test grid.z.Δᵃᵃᶜ == 1
    @test grid.z.Δᵃᵃᶠ == 1

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

    grid = LatitudeLongitudeGrid(CPU(), FT, size=(Nλ, Nφ, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test topology(grid) == (Periodic, Bounded, Bounded)

    @test grid.Nx == Nλ
    @test grid.Ny == Nφ
    @test grid.Nz == 1

    @test grid.Lx == 360
    @test grid.Ly == 160
    @test grid.Lz == 1

    @test grid.Δλᶠᵃᵃ == 10
    @test grid.Δφᵃᶠᵃ == 5
    @test grid.z.Δᵃᵃᶜ == 1
    @test grid.z.Δᵃᵃᶠ == 1

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

    (Nλ, Nφ, Nz) = grid_size = (24, 16, 16)
    (Hλ, Hφ, Hz) = halo      = ( 1,  1,  1)

    lat = (-80,   80)
    lon = (-180, 180)
    zᵣ  = (-100,   0)

    Λ₁  = (lat[1], lon[1], zᵣ[1])
    Λₙ  = (lat[2], lon[2], zᵣ[2])

    (Lλ, Lφ, Lz) = L = @. Λₙ - Λ₁

    grid_reg = LatitudeLongitudeGrid(CPU(), FT, size=grid_size, halo=halo, latitude=lat, longitude=lon, z=zᵣ)

    @test typeof(grid_reg.z.Δᵃᵃᶜ) == typeof(grid_reg.z.Δᵃᵃᶠ) == FT

    @test all(xspacings(grid_reg, Center(), Center()) .== reshape(grid_reg.Δxᶜᶜᵃ[1:Nφ],   1, Nφ,   1))
    @test all(xspacings(grid_reg, Center(), Face()  ) .== reshape(grid_reg.Δxᶜᶠᵃ[1:Nφ+1], 1, Nφ+1, 1))
    @test all(xspacings(grid_reg, Face(),   Center()) .== reshape(grid_reg.Δxᶠᶜᵃ[1:Nφ],   1, Nφ,   1))
    @test all(xspacings(grid_reg, Face(),   Face())   .== reshape(grid_reg.Δxᶠᶠᵃ[1:Nφ+1], 1, Nφ+1, 1))
    @test all(yspacings(grid_reg, Center(), Face()) .== grid_reg.Δyᶜᶠᵃ)
    @test all(yspacings(grid_reg, Face(),   Center()) .== grid_reg.Δyᶠᶜᵃ)
    @test all(zspacings(grid_reg, Center()) .== grid_reg.z.Δᵃᵃᶜ)
    @test all(zspacings(grid_reg, Face()) .== grid_reg.z.Δᵃᵃᶠ)

    @test all(xspacings(grid_reg, Center(), Center(), Center()) .== xspacings(grid_reg, Center(), Center()))
    @test all(xspacings(grid_reg, Face(),   Face(),   Center()) .== xspacings(grid_reg, Face(),   Face()))
    @test all(yspacings(grid_reg, Center(), Face(),   Center()) .== yspacings(grid_reg, Center(), Face()))
    @test all(yspacings(grid_reg, Face(),   Center(), Center()) .== yspacings(grid_reg, Face(),   Center()))
    @test all(zspacings(grid_reg, Face(),   Center(), Center()) .== zspacings(grid_reg, Center()))
    @test all(zspacings(grid_reg, Face(),   Center(), Face()  ) .== zspacings(grid_reg, Face()))

    @test Operators.Δx(1, 2, 3, grid_reg, Center(), Center(), Center()) == grid_reg.Δxᶜᶜᵃ[2]
    @test Operators.Δx(1, 2, 3, grid_reg, Center(), Face(),   Center()) == grid_reg.Δxᶜᶠᵃ[2]
    @test Operators.Δy(1, 2, 3, grid_reg, Center(), Face(),   Center()) == grid_reg.Δyᶜᶠᵃ
    @test Operators.Δy(1, 2, 3, grid_reg, Face(),   Center(), Center()) == grid_reg.Δyᶠᶜᵃ
    @test Operators.Δz(1, 2, 3, grid_reg, Center(), Center(), Face()  ) == grid_reg.z.Δᵃᵃᶠ
    @test Operators.Δz(1, 2, 3, grid_reg, Center(), Center(), Center()) == grid_reg.z.Δᵃᵃᶜ

    @test all(λspacings(grid_reg, Center()) .== grid_reg.Δλᶜᵃᵃ)
    @test all(λspacings(grid_reg, Face()) .== grid_reg.Δλᶠᵃᵃ)
    @test all(φspacings(grid_reg, Center()) .== grid_reg.Δφᵃᶜᵃ)
    @test all(φspacings(grid_reg, Face()) .== grid_reg.Δφᵃᶠᵃ)

    @test Operators.Δλ(1, 2, 3, grid_reg, Face(),   Center(), Face())   == grid_reg.Δλᶠᵃᵃ
    @test Operators.Δφ(1, 2, 3, grid_reg, Center(), Face(),   Center()) == grid_reg.Δφᵃᶠᵃ

    Δλ = grid_reg.Δλᶠᵃᵃ
    λₛ = (-grid_reg.Lx/2):Δλ:(grid_reg.Lx/2)

    Δz = grid_reg.z.Δᵃᵃᶜ
    zₛ = -Lz:Δz:0

    grid_str = LatitudeLongitudeGrid(CPU(), FT, size=grid_size, halo=halo, latitude=lat, longitude=λₛ, z=zₛ)

    @test length(grid_str.λᶠᵃᵃ) == length(grid_reg.λᶠᵃᵃ) == Nλ + 2Hλ
    @test length(grid_str.λᶜᵃᵃ) == length(grid_reg.λᶜᵃᵃ) == Nλ + 2Hλ

    @test length(grid_str.φᵃᶠᵃ) == length(grid_reg.φᵃᶠᵃ) == Nφ + 2Hφ + 1
    @test length(grid_str.φᵃᶜᵃ) == length(grid_reg.φᵃᶜᵃ) == Nφ + 2Hφ

    @test length(grid_str.z.cᵃᵃᶠ) == length(grid_reg.z.cᵃᵃᶠ) == Nz + 2Hz + 1
    @test length(grid_str.z.cᵃᵃᶜ) == length(grid_reg.z.cᵃᵃᶜ) == Nz + 2Hz

    @test length(grid_str.z.Δᵃᵃᶠ) == Nz + 2Hz + 1
    @test length(grid_str.z.Δᵃᵃᶜ) == Nz + 2Hz

    @test all(grid_str.λᶜᵃᵃ == grid_reg.λᶜᵃᵃ)
    @test all(grid_str.λᶠᵃᵃ == grid_reg.λᶠᵃᵃ)
    @test all(grid_str.φᵃᶜᵃ == grid_reg.φᵃᶜᵃ)
    @test all(grid_str.φᵃᶠᵃ == grid_reg.φᵃᶠᵃ)
    @test all(grid_str.z.cᵃᵃᶜ == grid_reg.z.cᵃᵃᶜ)
    @test all(grid_str.z.cᵃᵃᶠ == grid_reg.z.cᵃᵃᶠ)

    @test sum(grid_str.z.Δᵃᵃᶜ) == grid_reg.z.Δᵃᵃᶜ * length(grid_str.z.Δᵃᵃᶜ)
    @test sum(grid_str.z.Δᵃᵃᶠ) == grid_reg.z.Δᵃᵃᶠ * length(grid_str.z.Δᵃᵃᶠ)

    @test all(xspacings(grid_str, Center(), Center()) .== reshape(grid_str.Δxᶜᶜᵃ[1:Nλ, 1:Nφ],   Nλ, Nφ,   1))
    @test all(xspacings(grid_str, Center(), Face())   .== reshape(grid_str.Δxᶜᶠᵃ[1:Nλ, 1:Nφ+1], Nλ, Nφ+1, 1))
    @test all(xspacings(grid_str, Face(),   Center()) .== reshape(grid_str.Δxᶠᶜᵃ[1:Nλ, 1:Nφ],   Nλ, Nφ,   1))
    @test all(xspacings(grid_str, Face(),   Face())   .== reshape(grid_str.Δxᶠᶠᵃ[1:Nλ, 1:Nφ+1], Nλ, Nφ+1, 1))

    @test all(yspacings(grid_str, Center(), Face())   .== grid_str.Δyᶜᶠᵃ)
    @test all(yspacings(grid_str, Face(),   Center()) .== grid_str.Δyᶠᶜᵃ)

    @test all(zspacings(grid_str, Center()) .== reshape(grid_str.z.Δᵃᵃᶜ[1:Nz], 1, 1, Nz))
    @test all(zspacings(grid_str, Face()) .== reshape(grid_str.z.Δᵃᵃᶠ[1:Nz+1], 1, 1, Nz+1))

    @test all(zspacings(grid_str, Center()) .== zspacings(grid_str, Center(), Center(),   Center()))
    @test all(zspacings(grid_str, Face()) .== zspacings(grid_str, Face(), Center(), Face()))

    return nothing
end

function test_lat_lon_areas(FT)
    Nλ = 36
    Nφ = 32
    Hλ = Hφ = 2

    grid = LatitudeLongitudeGrid(CPU(), FT, size=(Nλ, Nφ, 1), longitude=(-180, 180), latitude=(-90, 90), z=(0, 1), halo=(Hλ, Hφ, 1))

    @test sum(grid.Azᶜᶜᵃ[1:grid.Ny]) * grid.Nx ≈ 4π * grid.radius^2

    return nothing
end

function test_lat_lon_xyzλφ_node_nodes(FT, arch)

    @info "    Testing with $FT on $(typeof(arch))..."

    (Nλ, Nφ, Nz) = grid_size = (12, 4, 2)
    (Hλ, Hφ, Hz) = halo      = (1, 1, 1)

    lat = (-60,   60)
    lon = (-180, 180)
    zᵣ  = (-10,    0)

    grid = LatitudeLongitudeGrid(CPU(), FT, size=grid_size, halo=halo, latitude=lat, longitude=lon, z=zᵣ)
    ibg  = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> y < 20 && y > -20 ? -50 : -0))

    @info "        Testing grid utils on LatitudeLongitude grid...."

    @test λnode(3, 1, 2, grid, Face(), Face(), Face()) ≈ -120
    @test φnode(3, 2, 2, grid, Face(), Face(), Face()) ≈ -30
    @test xnode(5, 1, 2, grid, Face(), Face(), Face()) / grid.radius ≈ -FT(π/6)
    @test ynode(2, 1, 2, grid, Face(), Face(), Face()) / grid.radius ≈ -FT(π/3)
    @test znode(2, 1, 2, grid, Face(), Face(), Face()) ≈ -5

    @test λnode(3, 1, 2, ibg, Face(), Face(), Face()) ≈ -120
    @test φnode(3, 2, 2, ibg, Face(), Face(), Face()) ≈ -30
    @test xnode(5, 1, 2, ibg, Face(), Face(), Face()) / ibg.radius ≈ -FT(π/6)
    @test ynode(2, 1, 2, ibg, Face(), Face(), Face()) / ibg.radius ≈ -FT(π/3)
    @test znode(2, 1, 2, ibg, Face(), Face(), Face()) ≈ -5

    @test minimum_xspacing(grid, Face(), Face(), Face()) / grid.radius ≈ FT(π/6) * cosd(60)
    @test minimum_xspacing(grid) / grid.radius ≈ FT(π/6) * cosd(45)
    @test minimum_yspacing(grid) / grid.radius ≈ FT(π/6)
    @test minimum_zspacing(grid) ≈ 5

    @test minimum_xspacing(ibg, Face(), Face(), Face()) / ibg.radius ≈ FT(π/6) * cosd(30)
    @test minimum_xspacing(ibg) / ibg.radius ≈ FT(π/6) * cosd(15)

    return nothing
end

function test_lat_lon_precomputed_metrics(FT, arch)
    Nλ, Nφ, Nz = N = (4, 2, 3)
    Hλ, Hφ, Hz = H = (1, 1, 1)

    latreg  = (-80,   80)
    lonreg  = (-180, 180)
    lonregB = (-160, 160)

    zreg    = (-1,     0)

    latstr  = [-80, 0, 80]
    lonstr  = [-180, -30, 10, 40, 180]
    lonstrB = [-160, -30, 10, 40, 160]
    zstr    = collect(0:Nz)

    latitude  = (latreg, latstr)
    longitude = (lonreg, lonstr, lonregB, lonstrB)
    zcoord    = (zreg,   zstr)

    CUDA.allowscalar() do

    # grid with pre computed metrics vs metrics computed on the fly
    for lat in latitude
        for lon in longitude
            for z in zcoord
                # println("$lat, $lon, $z")
                grid_pre = LatitudeLongitudeGrid(arch, FT, size=N, halo=H, latitude=lat, longitude=lon, z=z, precompute_metrics=true)
                grid_fly = LatitudeLongitudeGrid(arch, FT, size=N, halo=H, latitude=lat, longitude=lon, z=z)

                @test all(Array([all(Array([Δxᶠᶜᵃ(i, j, 1, grid_pre) ≈ Δxᶠᶜᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Δxᶜᶠᵃ(i, j, 1, grid_pre) ≈ Δxᶜᶠᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Δxᶠᶠᵃ(i, j, 1, grid_pre) ≈ Δxᶠᶠᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Δxᶜᶜᵃ(i, j, 1, grid_pre) ≈ Δxᶜᶜᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Δyᶜᶠᵃ(i, j, 1, grid_pre) ≈ Δyᶜᶠᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Azᶠᶜᵃ(i, j, 1, grid_pre) ≈ Azᶠᶜᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Azᶜᶠᵃ(i, j, 1, grid_pre) ≈ Azᶜᶠᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Azᶠᶠᵃ(i, j, 1, grid_pre) ≈ Azᶠᶠᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
                @test all(Array([all(Array([Azᶜᶜᵃ(i, j, 1, grid_pre) ≈ Azᶜᶜᵃ(i, j, 1, grid_fly) for i in 1-Hλ+1:Nλ+Hλ-1])) for j in 1-Hφ+1:Nφ+Hφ-1]))
            end
        end
    end

    end # CUDA.allowscalar()

end

#####
##### Conformal cubed sphere face grid
#####

function test_orthogonal_shell_grid_array_sizes_and_spacings(FT)

    grid = ConformalCubedSpherePanelGrid(CPU(), FT, size=(10, 10, 1), z=(0, 1))

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

    @test all(xspacings(grid, Center(), Center(), Face()) .== xspacings(grid, Center(), Center()) .== grid.Δxᶜᶜᵃ[1:Nx,   1:Ny])
    @test all(xspacings(grid, Center(), Face(),   Face()) .== xspacings(grid, Center(), Face()  ) .== grid.Δxᶜᶠᵃ[1:Nx,   1:Ny+1])
    @test all(xspacings(grid, Face(),   Center(), Face()) .== xspacings(grid, Face(),   Center()) .== grid.Δxᶠᶜᵃ[1:Nx+1, 1:Ny])
    @test all(xspacings(grid, Face(),   Face(),   Face()) .== xspacings(grid, Face(),   Face()  ) .== grid.Δxᶠᶠᵃ[1:Nx+1, 1:Ny+1])

    @test all(yspacings(grid, Center(), Center(), Face()) .== yspacings(grid, Center(), Center()) .== grid.Δyᶜᶜᵃ[1:Nx,   1:Ny])
    @test all(yspacings(grid, Center(), Face(),   Face()) .== yspacings(grid, Center(), Face()  ) .== grid.Δyᶜᶠᵃ[1:Nx,   1:Ny+1])
    @test all(yspacings(grid, Face(),   Center(), Face()) .== yspacings(grid, Face(),   Center()) .== grid.Δyᶠᶜᵃ[1:Nx+1, 1:Ny])
    @test all(yspacings(grid, Face(),   Face(),   Face()) .== yspacings(grid, Face(),   Face()  ) .== grid.Δyᶠᶠᵃ[1:Nx+1, 1:Ny+1])

    @test all(zspacings(grid, Center(), Center(), Face()  ) .== zspacings(grid, Face()  ) .== grid.z.Δᵃᵃᶠ)
    @test all(zspacings(grid, Center(), Center(), Center()) .== zspacings(grid, Center()) .== grid.z.Δᵃᵃᶜ)

    return nothing
end


#####
##### Test the tests
#####

@testset "Grids" begin
    @info "Testing AbstractGrids..."

    @testset "Grid utils" begin
        @info "  Testing grid utilities..."
        @test total_extent(Periodic(), 1, 0.2, 1.0) == 1.2
        @test total_extent(Bounded(), 1, 0.2, 1.0) == 1.4
    end

    @testset "Coordinate utils" begin
        @info "  Testing ExponentialCoordinate..."

        for arch in archs
            Nx = 10
            l, r = -1000, 100
            scale = (r - l) / 5

            xₗ = ExponentialCoordinate(Nx, l, r; scale, bias =:left)
            xᵣ = ExponentialCoordinate(Nx, l, r; scale, bias =:right)

            for i in 1:Nx+1
                @test xᵣ[i] == xᵣ(i)
                @test xₗ[i] == xₗ(i)
            end

            @test length(xₗ) == Nx
            @test xₗ(1) == l
            @test xₗ(Nx+1) == r
            @test xᵣ(1) == l
            @test xᵣ(Nx+1) == r
            @test xₗ(Nx+1) - xₗ(Nx) ≈ xᵣ(2) - xᵣ(1)
            @test xᵣ(Nx+1) - xᵣ(Nx) ≈ xₗ(2) - xₗ(1)

            for x in (xₗ, xᵣ)
                grid = RectilinearGrid(arch; size=(Nx, 4), x, z=(-1, 0), topology=(Bounded, Flat, Bounded))
                for i in 1:Nx+1
                    @allowscalar @test grid.xᶠᵃᵃ[i] == x(i)
                end
            end

            @info "  Testing ConstantToStretchedCoordinate..."
            extent = 200
            constant_spacing = 25
            constant_spacing_extent = 90
            z = ConstantToStretchedCoordinate(; extent, constant_spacing, constant_spacing_extent)

            Nz = length(z)

            @test length(z.faces) == Nz+1

            Δz = diff(z.faces)

            N_uniform_cells = Int(ceil(constant_spacing_extent / constant_spacing))

            for k in 1:Nz-N_uniform_cells
                @test Δz[k] > constant_spacing
            end
            for k in Nz-(N_uniform_cells-1):Nz
                @test Δz[k] == constant_spacing
            end

            grid = RectilinearGrid(arch; size=length(z), z, topology=(Flat, Flat, Bounded))
            @test grid.z.cᵃᵃᶠ[1:Nz+1] == on_architecture(arch, z.faces)
            @test grid.z.Δᵃᵃᶜ[1:Nz] == on_architecture(arch, Δz)

            # kwarg values that give a uniformly-spaced coordinate
            Nz = 7
            constant_spacing = 25.34
            constant_spacing_extent = Nz * constant_spacing
            extent = constant_spacing_extent
            z = ConstantToStretchedCoordinate(; extent, constant_spacing, constant_spacing_extent)

            @test length(z) == Nz
            @test length(z.faces) == Nz+1

            Δz = diff(z.faces)
            @test all(Δz .≈ constant_spacing)
            @test z(Nz+1) - z(1) ≈ extent

            grid = RectilinearGrid(arch; size=length(z), z, topology=(Flat, Flat, Bounded))
            @test grid.z.cᵃᵃᶠ[1:Nz+1] == on_architecture(arch, z.faces)
            @test grid.z.Δᵃᵃᶜ[1:Nz] == on_architecture(arch, Δz)
        end
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
                for arch in archs
                    test_regular_rectilinear_xnode_ynode_znode_and_spacings(arch, FT)
                end
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

        @testset "Grid equality" begin
            @info "    Testing grid equality operator (==)..."

            for arch in archs
                test_grid_equality(arch)
                test_immersed_boundary_grid_equality(arch)
            end

            if CUDA.has_cuda()
                test_grid_equality_over_architectures()
            end
        end

        # Testing show function
        topo = (Periodic, Periodic, Periodic)

        grid = RectilinearGrid(CPU(), topology=topo, size=(3, 7, 9), x=(0, 1), y=(-π, π), z=(0, 2π))

        @test try
            show(grid); println()
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
                for N in [16, 17]
                    test_rectilinear_grid_correct_spacings(FT, N)
                end
            end

            # Testing show function
            Nz = 20
            grid = RectilinearGrid(arch, size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=collect(0:Nz).^2)

            @test try
                show(grid); println()
                true
            catch err
                println("error in show(::RectilinearGrid)")
                println(sprint(showerror, err))
                false
            end

            @test grid isa RectilinearGrid
        end

        for arch in archs
            @info "  Testing on_architecture for RectilinearGrid..."
            cpu_grid = RectilinearGrid(CPU(), size=(1, 1, 4), x=(0, 1), y=(0, 1), z=collect(0:4).^2)
            grid = on_architecture(arch, cpu_grid)
            @test grid isa RectilinearGrid
            @test architecture(grid) == arch
            cpu_grid_again = on_architecture(CPU(), grid)
            @test cpu_grid_again == cpu_grid
        end
    end

    @testset "Latitude-longitude grid" begin
        @info "  Testing latitude-longitude grid construction errors..."

	for FT in float_types
	    @test_throws ArgumentError LatitudeLongitudeGrid(CPU(), FT, size=(10, 10, 4), longitude=(-180, 180), latitude=(-80, 80), z=[-50.0, -30.0, -20.0, 0.0]) # too few z-faces
	    @test_throws ArgumentError LatitudeLongitudeGrid(CPU(), FT, size=(10, 10, 4), longitude=(-180, 180), latitude=(-80, 80), z=[-2000.0, -1000.0, -50.0, -30.0, -20.0, 0.0]) # too many z-faces
	end

        @info "  Testing general latitude-longitude grid..."

        for FT in float_types
            test_basic_lat_lon_bounded_domain(FT)
            test_basic_lat_lon_periodic_domain(FT)
            test_basic_lat_lon_general_grid(FT)
            test_lat_lon_areas(FT)
        end

        @info "  Testing precomputed metrics on LatitudeLongitudeGrid..."
        for arch in archs, FT in float_types
            test_lat_lon_precomputed_metrics(FT, arch)
            test_lat_lon_xyzλφ_node_nodes(FT, arch)
        end

        # Testing show function for regular grid
        grid = LatitudeLongitudeGrid(CPU(), size=(36, 32, 1), longitude=(-180, 180), latitude=(-80, 80), z=(0, 1))

        @test try
            show(grid); println()
            true
        catch err
            println("error in show(::LatitudeLongitudeGrid)")
            println(sprint(showerror, err))
            false
        end

        @test grid isa LatitudeLongitudeGrid

        for arch in archs
            @info "  Testing show for vertically-stretched LatitudeLongitudeGrid..."
            grid = LatitudeLongitudeGrid(arch,
                                         size = (36, 32, 10),
                                         longitude = (-180, 180),
                                         latitude = (-80, 80),
                                         z = collect(0:10))

            @test try
                show(grid); println()
                true
            catch err
                println("error in show(::LatitudeLongitudeGrid)")
                println(sprint(showerror, err))
                false
            end

            @test grid isa LatitudeLongitudeGrid
        end

        for arch in archs
            @info "  Testing on_architecture for LatitudeLongitudeGrid..."
            cpu_grid = LatitudeLongitudeGrid(CPU(),
                                             size = (36, 32, 10),
                                             longitude = (-180, 180),
                                             latitude = (-80, 80),
                                             z = collect(0:10))
            grid = on_architecture(arch, cpu_grid)
            @test grid isa LatitudeLongitudeGrid
            @test architecture(grid) == arch

            cpu_grid_again = on_architecture(CPU(), grid)
            @test cpu_grid_again == cpu_grid
        end

        for arch in archs
            for FT in float_types
                @info " Testing construction of LatitudeLongitudeGrid from RectilinearGrid..."
                Nx = 4
                Ny = 5
                Nz = 6
                Lx = Ly = 2π / 180
                rectilinear_grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(0, 1), topology=(Bounded, Bounded, Bounded))
                lat_lon_grid = LatitudeLongitudeGrid(rectilinear_grid, radius=1, origin=(0, 0))
                λ = λnodes(lat_lon_grid, Face())
                φ = φnodes(lat_lon_grid, Face())
                @test λ[1]  == -1
                @test λ[Nx+1] == +1
                @test φ[1]  == -1
                @test φ[Ny+1] == +1
                @test znodes(rectilinear_grid, Center()) == znodes(lat_lon_grid, Center())
                @test znodes(rectilinear_grid, Face()) == znodes(lat_lon_grid, Face())

                lat_lon_grid = LatitudeLongitudeGrid(rectilinear_grid, radius=1, origin=(273, 0))
                λ = λnodes(lat_lon_grid, Face())
                φ = φnodes(lat_lon_grid, Face())
                @test λ[1]  == 272
                @test λ[Nx+1] == 274
                @test φ[1]  == -1
                @test φ[Ny+1] == +1
                @test znodes(rectilinear_grid, Center()) == znodes(lat_lon_grid, Center())
                @test znodes(rectilinear_grid, Face()) == znodes(lat_lon_grid, Face())


                Lx = 2π / 180 * cosd(45)
                Ly = 2π / 180
                rectilinear_grid = RectilinearGrid(arch, FT, size=(Nx, Ny, Nz), x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(0, 1), topology=(Bounded, Bounded, Bounded))
                lat_lon_grid = LatitudeLongitudeGrid(rectilinear_grid, radius=1, origin=(0, 45))

                λ = λnodes(lat_lon_grid, Face())
                φ = φnodes(lat_lon_grid, Face())
                @test λ[1]  == -1
                @test λ[Nx+1] == +1
                @test φ[1]  == 44
                @test φ[Ny+1] == 46
                @test znodes(rectilinear_grid, Center()) == znodes(lat_lon_grid, Center())
                @test znodes(rectilinear_grid, Face()) == znodes(lat_lon_grid, Face())
            end
        end
    end

    @testset "Single column grids" begin
        @info "  Testing single column grid construction..."

        for arch in archs
            for FT in float_types
                ccc = (Center(), Center(), Center())
                grid = RectilinearGrid(arch, FT, size=4, z=(-1, 0), topology=(Flat, Flat, Bounded))
                x = xnodes(grid, ccc...)
                y = ynodes(grid, ccc...)
                @test isnothing(x)
                @test isnothing(y)

                x₀ = 1
                y₀ = π
                grid = RectilinearGrid(arch, FT, size=4, x=x₀, y=y₀, z=(-1, 0), topology=(Flat, Flat, Bounded))
                x = xnodes(grid, ccc...)
                y = ynodes(grid, ccc...)
                @test x[1] isa FT
                @test y[1] isa FT
                @test x[1] == x₀
                @test y[1] == convert(FT, y₀)

                grid = LatitudeLongitudeGrid(arch, FT, size=4, z=(-1, 0), topology=(Flat, Flat, Bounded))
                λ = λnodes(grid, ccc...)
                φ = φnodes(grid, ccc...)
                @test isnothing(λ)
                @test isnothing(φ)

                λ₀ = 45
                φ₀ = 10.1
                grid = LatitudeLongitudeGrid(arch, FT, size=4, latitude=φ₀, longitude=λ₀, z=(-1, 0),
                                             topology=(Flat, Flat, Bounded))
                λ = λnodes(grid, ccc...)
                φ = φnodes(grid, ccc...)
                @test λ[1] isa FT
                @test φ[1] isa FT
                @test λ[1] == λ₀
                @test φ[1] == convert(FT, φ₀)

                halo_sz = ColumnEnsembleSize(; Nz=4, ensemble=(2, 3), Hz=5)
                grid = RectilinearGrid(arch, FT; size=halo_sz, halo=halo_sz,
                                       z=(-10, 0), topology=(Flat, Flat, Bounded))
                @test size(grid) == (2, 3, 4)
                @test halo_size(grid) == (0, 0, 5)

                if arch isa GPU
                    cpu_grid = on_architecture(CPU(), grid)
                    @test size(cpu_grid) == (2, 3, 4)
                    @test halo_size(cpu_grid) == (0, 0, 5)

                    gpu_grid = on_architecture(GPU(), grid)
                    @test gpu_grid == grid
                end
            end
        end
    end

    @testset "Conformal cubed sphere face grid" begin
        @info "  Testing OrthogonalSphericalShellGrid grid..."

        for FT in float_types
            test_orthogonal_shell_grid_array_sizes_and_spacings(FT)
        end

        # Testing show function
        grid = ConformalCubedSpherePanelGrid(CPU(), size=(10, 10, 1), z=(0, 1))

        @test try
            show(grid); println()
            true
        catch err
            println("error in show(::OrthogonalSphericalShellGrid)")
            println(sprint(showerror, err))
            false
        end

        @test grid isa OrthogonalSphericalShellGrid

        grid = RotatedLatitudeLongitudeGrid(size = (10, 10, 1),
                                            latitude = (-60, 60),
                                            longitude = (-60, 60),
                                            z = (-1000, 0),
                                            north_pole = (0, 0),
                                            topology = (Bounded, Bounded, Bounded))

        @show grid.conformal_mapping

        @test grid isa OrthogonalSphericalShellGrid
        @test grid isa RotatedLatitudeLongitudeGrid
        @test grid.Lz == 1000
        @test size(grid) == (10, 10, 1)

        for arch in archs
            for FT in float_types
                z = (0, 1)
                radius = 234.5e6

                Nx, Ny = 10, 8
                grid = ConformalCubedSpherePanelGrid(arch, FT, size=(Nx, Ny, 1); z, radius)

                # the sum of area metrics Azᶜᶜᵃ is 1/6-th of the area of the sphere
                @test sum(grid.Azᶜᶜᵃ[1:Nx, 1:Ny]) ≈ 4π * grid.radius^2 / 6

                # the sum of the distance metrics Δxᶜᶜᵃ and Δyᶜᶜᵃ that correspond to great circles
                # are 1/4-th of the circumference of the sphere's great circle

                # (for odd number of grid points, the central grid points fall on great circles)
                Nx, Ny = 11, 9
                grid = ConformalCubedSpherePanelGrid(arch, FT, size=(Nx, Ny, 1); z, radius)
                @test sum(grid.Δxᶜᶜᵃ[1:Nx, (Ny+1)÷2]) ≈ 2π * grid.radius / 4
                @test sum(grid.Δyᶜᶜᵃ[(Nx+1)÷2, 1:Ny]) ≈ 2π * grid.radius / 4

                Nx, Ny = 10, 9
                grid = ConformalCubedSpherePanelGrid(arch, FT, size=(Nx, Ny, 1); z, radius)
                @test sum(grid.Δxᶜᶜᵃ[1:Nx, (Ny+1)÷2]) ≈ 2π * grid.radius / 4

                Nx, Ny = 11, 8
                grid = ConformalCubedSpherePanelGrid(arch, FT, size=(Nx, Ny, 1); z, radius)
                @test sum(grid.Δyᶜᶜᵃ[(Nx+1)÷2, 1:Ny]) ≈ 2π * grid.radius / 4
            end
        end
    end
end
