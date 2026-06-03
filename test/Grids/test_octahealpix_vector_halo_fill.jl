using Test
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnodes, φnodes

# Verify the OctaHEALPix paired vector halo fill writes nonzero, idempotent,
# topology-consistent values into the (u, v) halos. This locks in the recent
# `_fill_octahealpix_u_vector_halos!` / `_fill_octahealpix_v_vector_halos!`
# fix and prevents the `= nothing` no-op regression.

function setup_constant_uv(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)

    u_value = convert(eltype(grid), 11//10)
    v_value = convert(eltype(grid), -23//10)

    for j in 1:grid.Ny, i in 1:(grid.Nx + 1)
        u[i, j, 1] = u_value
    end

    for j in 1:(grid.Ny + 1), i in 1:grid.Nx
        v[i, j, 1] = v_value
    end

    fill_halo_regions!((u, v))
    return u, v, u_value, v_value
end

@testset "OctaHEALPix paired vector halo fill" begin
    for FT in (Float32, Float64)
        grid = SphericalShellGrid(CPU(), FT;
                                  mapping = OctaHEALPixMapping(8),
                                  z = (zero(FT), one(FT)),
                                  radius = one(FT),
                                  halo = (5, 5, 3))

        Nx, Ny = grid.Nx, grid.Ny
        Hx, Hy = grid.Hx, grid.Hy

        u, v, u_value, v_value = setup_constant_uv(grid)

        # 1. halos are not literally zero — defends the `= nothing` regression
        west_u = [u[1 - i, j, 1] for j in 1:Ny, i in 1:Hx]
        east_u = [u[Nx + 1 + i, j, 1] for j in 1:Ny, i in 1:Hx]
        south_v = [v[i, 1 - j, 1] for j in 1:Hy, i in 1:Nx]
        north_v = [v[i, Ny + 1 + j, 1] for j in 1:Hy, i in 1:Nx]

        @test maximum(abs, west_u) > zero(FT)
        @test maximum(abs, east_u) > zero(FT)
        @test maximum(abs, south_v) > zero(FT)
        @test maximum(abs, north_v) > zero(FT)

        # 2. halo magnitudes are bounded by the interior input (no amplification)
        bound = max(abs(u_value), abs(v_value)) + 10eps(FT)
        for j in 1:Ny, i in 1:Hx
            @test abs(u[1 - i, j, 1])     ≤ bound
            @test abs(u[Nx + 1 + i, j, 1]) ≤ bound
        end
        for j in 1:Hy, i in 1:Nx
            @test abs(v[i, 1 - j, 1])     ≤ bound
            @test abs(v[i, Ny + 1 + j, 1]) ≤ bound
        end

        # 3. idempotence: calling fill_halo_regions! again must not change halos
        u_snapshot = copy(parent(u.data))
        v_snapshot = copy(parent(v.data))
        fill_halo_regions!((u, v))
        @test parent(u.data) == u_snapshot
        @test parent(v.data) == v_snapshot

        # 4. interior is untouched by halo fill
        for j in 1:Ny, i in 1:(Nx + 1)
            @test u[i, j, 1] == u_value
        end
        for j in 1:(Ny + 1), i in 1:Nx
            @test v[i, j, 1] == v_value
        end

        # 5. consistency with the connectivity rotation: each halo value must
        #    equal the rotated source value via
        #    `octahealpix_xface_vector_halo_source` / `octahealpix_yface_vector_halo_source`.
        connectivity = grid.connectivity

        for j in 1:Ny, i in (1 - Hx):0
            source_kind, source_i, source_j, sign_factor =
                Oceananigans.Grids.octahealpix_xface_vector_halo_source(
                    i, j, Nx, Ny, connectivity, Val(:covariant))
            expected = sign_factor * (source_kind == 1 ?
                                      u[source_i, source_j, 1] :
                                      v[source_i, source_j, 1])
            @test u[i, j, 1] ≈ expected atol = 10eps(FT)
        end

        for j in 1:Ny, i in (Nx + 2):(Nx + 1 + Hx)
            source_kind, source_i, source_j, sign_factor =
                Oceananigans.Grids.octahealpix_xface_vector_halo_source(
                    i, j, Nx, Ny, connectivity, Val(:covariant))
            expected = sign_factor * (source_kind == 1 ?
                                      u[source_i, source_j, 1] :
                                      v[source_i, source_j, 1])
            @test u[i, j, 1] ≈ expected atol = 10eps(FT)
        end

        for j in (1 - Hy):0, i in 1:Nx
            source_kind, source_i, source_j, sign_factor =
                Oceananigans.Grids.octahealpix_yface_vector_halo_source(
                    i, j, Nx, Ny, connectivity, Val(:covariant))
            expected = sign_factor * (source_kind == 1 ?
                                      u[source_i, source_j, 1] :
                                      v[source_i, source_j, 1])
            @test v[i, j, 1] ≈ expected atol = 10eps(FT)
        end

        for j in (Ny + 2):(Ny + 1 + Hy), i in 1:Nx
            source_kind, source_i, source_j, sign_factor =
                Oceananigans.Grids.octahealpix_yface_vector_halo_source(
                    i, j, Nx, Ny, connectivity, Val(:covariant))
            expected = sign_factor * (source_kind == 1 ?
                                      u[source_i, source_j, 1] :
                                      v[source_i, source_j, 1])
            @test v[i, j, 1] ≈ expected atol = 10eps(FT)
        end
    end
end
