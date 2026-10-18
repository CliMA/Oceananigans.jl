include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Grids: FullyConnected
using Oceananigans.MultiRegion: CubedSpherePartition, CubedSphereConnectivity, MultiRegionGrid, MultiRegionObject,
                                getregion, synchronize_shared_edges!
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Run with --check-bounds=yes: the halo kernels use @inbounds, so only forced bounds checking catches storage overruns.

N, Nz = 4, 2
cubed_sphere(; H = 4) = ConformalCubedSphereGrid(; panel_size = (N, N, Nz), z = (0, 1), horizontal_direction_halo = H)

# ConformalCubedSphereGrid cannot yet build R > 1, so assemble a subdivided layout from equal regional panels.
function subdivided_cubed_sphere(R)
    partition = CubedSpherePartition(; R)
    topology = (FullyConnected, FullyConnected, Bounded)
    panels = Tuple(OrthogonalSphericalShellGrid(CPU(); size = (N, N, Nz), z = (0, 1), halo = (4, 4, 1), topology)
                   for _ in 1:length(partition))
    return MultiRegionGrid{Float64, topology..., typeof(panels[1].z)}(CPU(), partition, CubedSphereConnectivity(partition),
                                                                       MultiRegionObject(panels))
end

coded_u(r, i, j, k) = 10000r + 100i + j + k / 100
coded_v(r, i, j, k) = -(10000r + 100i + j + k / 100)

function coded_dgrid_pair(grid; LZ = Center)
    u = Field{Center, Face, LZ}(grid)
    v = Field{Face, Center, LZ}(grid)
    Nr = length(grid.partition)
    for r in 1:Nr
        du = getregion(u, r).data; dv = getregion(v, r).data
        fill!(parent(du), NaN); fill!(parent(dv), NaN)
        Nx, Ny, Nk = size(getregion(u, r))
        for k in 1:Nk, j in 1:Ny+1, i in 1:Nx
            du[i, j, k] = coded_u(r, i, j, k)
        end
        for k in 1:Nk, j in 1:Ny, i in 1:Nx+1
            dv[i, j, k] = coded_v(r, i, j, k)
        end
    end
    return u, v
end

snapshot(f) = [copy(parent(getregion(f, r).data)) for r in 1:length(f.grid.partition)]
unchanged(f, before) = all(isequal(parent(getregion(f, r).data), before[r]) for r in 1:length(f.grid.partition))

function rejects_without_writing(call, u, v)
    su, sv = snapshot(u), snapshot(v)
    threw = try
        call()
        false
    catch e
        e isa ArgumentError || rethrow()
        true
    end
    return threw && unchanged(u, su) && unchanged(v, sv)
end

@testset "D-grid exchange: valid six-panel layout" begin
    grid = cubed_sphere()
    H = 4
    u, v = coded_dgrid_pair(grid)
    fill_halo_regions!(u, v)  # default width H - 1
    for r in 1:6
        du = getregion(u, r).data; dv = getregion(v, r).data
        for k in 1:Nz, t in 1:N, d in 1:H-1
            @test isfinite(du[1 - d, t, k]) && isfinite(du[N + d, t, k]) && isfinite(du[t, 1 - d, k]) && isfinite(du[t, N + 1 + d, k])
            @test isfinite(dv[1 - d, t, k]) && isfinite(dv[N + 1 + d, t, k]) && isfinite(dv[t, 1 - d, k]) && isfinite(dv[t, N + d, k])
        end
        for k in 1:Nz, t in 1:N
            @test isnan(du[1 - H, t, k]) && isnan(du[t, 1 - H, k]) && isnan(dv[1 - H, t, k]) && isnan(dv[t, 1 - H, k])
            @test isnan(du[N + H, t, k]) && isnan(dv[t, N + H, k])
        end
        @test all(isnan, du[1-H:0, 1-H:0, 1:Nz]) && all(isnan, dv[N+2:N+H, N+1:N+H, 1:Nz])
    end
    # Unrotated east neighbour of region 1 is region 2; rotated north neighbour is region 3 (u ↦ -v).
    for k in 1:Nz, j in 1:N, d in 1:H-1
        @test getregion(u, 1)[N + d, j, k] === coded_u(2, d, j, k)
    end
    for k in 1:Nz, i in 1:N, d in 1:H-1
        @test getregion(u, 1)[i, N + 1 + d, k] === -coded_v(3, d + 1, N + 1 - i, k)
    end

    fill_halo_regions!(u, v; halo_width = H - 1)
    u, v = coded_dgrid_pair(grid)
    synchronize_shared_edges!(u, v)
    for k in 1:Nz, i in 1:N
        @test getregion(u, 1)[i, N + 1, k] === -coded_v(3, 1, N + 1 - i, k)
        @test getregion(v, 1)[N + 1, i, k] === coded_v(2, 1, i, k)
    end
end

@testset "D-grid exchange: fail-closed preflight" begin
    grid = cubed_sphere()
    u, v = coded_dgrid_pair(grid)
    @test rejects_without_writing(() -> fill_halo_regions!(u, v; halo_width = 4), u, v)
    @test rejects_without_writing(() -> fill_halo_regions!(u, v; halo_width = 0), u, v)

    shallow = cubed_sphere(; H = 1)
    us, vs = coded_dgrid_pair(shallow)
    @test rejects_without_writing(() -> fill_halo_regions!(us, vs), us, vs)
    @test rejects_without_writing(() -> fill_halo_regions!(us, vs; halo_width = 1), us, vs)

    subdivided = subdivided_cubed_sphere(2)
    ur, vr = coded_dgrid_pair(subdivided)
    @test rejects_without_writing(() -> fill_halo_regions!(ur, vr; halo_width = 1), ur, vr)
    @test rejects_without_writing(() -> synchronize_shared_edges!(ur, vr), ur, vr)

    other_grid = cubed_sphere()
    uo, vo = coded_dgrid_pair(other_grid)
    @test rejects_without_writing(() -> fill_halo_regions!(u, vo), u, vo)
    @test rejects_without_writing(() -> synchronize_shared_edges!(u, vo), u, vo)

    uf, vf = coded_dgrid_pair(grid; LZ = Face)
    @test rejects_without_writing(() -> fill_halo_regions!(u, vf), u, vf)
    @test rejects_without_writing(() -> synchronize_shared_edges!(u, vf), u, vf)

    windowed = Field{Face, Center, Center}(grid; indices = (:, :, 1:1))
    @test rejects_without_writing(() -> fill_halo_regions!(u, windowed), u, windowed)
    @test rejects_without_writing(() -> synchronize_shared_edges!(u, windowed), u, windowed)

    # Halo-inclusive and full-interior windows keep the logical size of a full field (R035-02 applies here too).
    for indices in ((-3:N+4, :, :), (:, :, 0:Nz+1), (:, :, 1:Nz))
        uw = Field{Center, Face, Center}(grid; indices)
        @test rejects_without_writing(() -> fill_halo_regions!(uw, v), uw, v)
        @test rejects_without_writing(() -> synchronize_shared_edges!(uw, v), uw, v)
    end
end
