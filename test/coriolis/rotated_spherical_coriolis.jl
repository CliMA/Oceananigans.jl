include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Coriolis: HydrostaticFormulation, NonhydrostaticFormulation,
                            TriadScheme, ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: volume
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

# The discrete Coriolis acceleration of a smooth velocity field is compared with the exact 2Ω × U in Cartesian
# coordinates, independently of the grid's rotation: the ŷ axis of a `RotatedLatitudeLongitudeGrid` points toward
# its north pole `P`, and a `LatitudeLongitudeGrid` is the case `P = ẑ`.

# The position on the unit sphere of the geographic point (λ, φ)
position(λ, φ) = [cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ)]

# The unit vectors (x̂, ŷ, ẑ) at `X` of a grid whose north pole is at `P`
function grid_basis(X, P)
    ŷ = normalize(P - (P ⋅ X) * X)
    return (ŷ × X, ŷ, X)
end

velocity(X) = [X[2] * X[3] + 2, X[1]^2 - X[3], X[1] * X[2] + 1]

# 2Ω × U, or under the traditional approximation f ẑ × U with the local vertical ẑ = X and f = 2Ω sin φ = 2Ω X[3]
coriolis_acceleration(X, U, Ω, traditional) = traditional ? 2Ω * X[3] * (X × U) : 2Ω * ([0, 0, 1] × U)

# Set `q` to the component along the grid's `axis` of the Cartesian vector field `vector(X)`, halos included
function set_component!(q, vector, P, axis)
    ℓx, ℓy, ℓz = map(ℓ -> ℓ(), location(q))
    for I in CartesianIndices(q.data)
        i, j, k = Tuple(I)
        X = position(λnode(i, j, k, q.grid, ℓx, ℓy, ℓz), φnode(i, j, k, q.grid, ℓx, ℓy, ℓz))
        q.data[I] = vector(X) ⋅ grid_basis(X, P)[axis]
    end
    return q
end

velocity_fields(grid, P) = (set_component!(XFaceField(grid), velocity, P, 1),
                            set_component!(YFaceField(grid), velocity, P, 2),
                            set_component!(ZFaceField(grid), velocity, P, 3))

function coriolis_accelerations(grid, coriolis, U)
    ax = Field(KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, grid, coriolis, U))
    ay = Field(KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, grid, coriolis, U))
    az = Field(KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, grid, coriolis, U))
    return compute!(ax), compute!(ay), compute!(az)
end

# The maximum error of the discrete Coriolis acceleration, relative to the largest exact acceleration
function coriolis_error(grid, P, coriolis, traditional)
    accelerations = coriolis_accelerations(grid, coriolis, velocity_fields(grid, P))
    exact(X) = coriolis_acceleration(X, velocity(X), coriolis.rotation_rate, traditional)
    exact_accelerations = map((a, axis) -> set_component!(similar(a), exact, P, axis), accelerations, (1, 2, 3))
    error = maximum(maximum(abs, interior(a) .- interior(e)) for (a, e) in zip(accelerations, exact_accelerations))
    return error / maximum(maximum(abs, interior(e)) for e in exact_accelerations)
end

@testset "SphericalCoriolis on rotated grids" begin
    arch = CPU() # the fields are set one point at a time
    north_pole = (0, 0)
    P = position(north_pole...)
    kw = (halo = (3, 3, 3), z = (0, 1), longitude = (40, 80), latitude = (-20, 20))
    rotated_grid(N) = RotatedLatitudeLongitudeGrid(arch; size = (N, N, 2), north_pole, kw...,
                                                   topology = (Bounded, Bounded, Bounded))
    latitude_longitude_grid(N) = LatitudeLongitudeGrid(arch; size = (N, N, 2), kw...)

    @testset "Convergence to the exact acceleration" begin
        # The active-weighted schemes are left out: their wet-point correction counts the wall faces of this
        # bounded grid as inactive, which is only consistent with a velocity that vanishes there.
        for (scheme, formulation) in ((EnstrophyConserving(), NonhydrostaticFormulation()),
                                      (TriadScheme(), NonhydrostaticFormulation()),
                                      (EnergyConserving(), HydrostaticFormulation()))
            coriolis = SphericalCoriolis(; scheme, formulation)
            traditional = formulation isa HydrostaticFormulation
            for (grid, pole) in ((rotated_grid, P), (latitude_longitude_grid, [0, 0, 1]))
                coarse, fine = map(N -> coriolis_error(grid(N), pole, coriolis, traditional), (8, 16))
                @test coarse < 1e-2
                @test coarse / fine > 3 # second order
            end
        end
    end

    @testset "Reduction to the hydrostatic formulation" begin
        # The nontraditional terms multiply w in the horizontal components and u, v in the vertical one, so with
        # w ≡ 0 every scheme's horizontal acceleration is its traditional stencil exactly, and with u ≡ v ≡ 0
        # the vertical acceleration vanishes, as it always does in the hydrostatic formulation.
        grid = rotated_grid(8)
        u, v, w = velocity_fields(grid, P)
        for scheme in (EnstrophyConserving(), EnergyConserving(), TriadScheme(),
                       ActiveWeightedEnstrophyConserving(), ActiveWeightedEnergyConserving())
            nonhydrostatic = SphericalCoriolis(; scheme)
            hydrostatic = HydrostaticSphericalCoriolis(; scheme)
            ax, ay, _ = coriolis_accelerations(grid, nonhydrostatic, (u, v, ZFaceField(grid)))
            ax_hydrostatic, ay_hydrostatic, az_hydrostatic = coriolis_accelerations(grid, hydrostatic, (u, v, w))
            @test interior(ax) == interior(ax_hydrostatic)
            @test interior(ay) == interior(ay_hydrostatic)
            @test all(iszero, interior(az_hydrostatic))
            _, _, az = coriolis_accelerations(grid, nonhydrostatic, (XFaceField(grid), YFaceField(grid), w))
            @test all(iszero, interior(az))
        end
    end

    @testset "Other orthogonal spherical shell grids" begin
        # Grids without a location-aware rotation angle use the cell-centre one. The two poles of the tripolar
        # grid lie on u-points of its top row, where the metrics vanish and no Coriolis stencil is finite.
        grid = TripolarGrid(arch; size = (16, 10, 2), z = (0, 1))
        U = (XFaceField(grid), YFaceField(grid), ZFaceField(grid))
        for (q, component) in zip(U, ((λ, φ, z) -> cosd(φ), (λ, φ, z) -> sind(λ) * cosd(φ), (λ, φ, z) -> cosd(λ)))
            set!(q, component)
            fill_halo_regions!(q)
        end

        nonhydrostatic = coriolis_accelerations(grid, SphericalCoriolis(), U)
        hydrostatic = coriolis_accelerations(grid, HydrostaticSphericalCoriolis(), U)
        for (a, aʰ) in zip(nonhydrostatic, hydrostatic)
            @test isfinite.(interior(a)) == isfinite.(interior(aʰ))
        end

        ax, ay, az = nonhydrostatic
        @test count(!isfinite, interior(ax)) == 2 * size(grid, 3)
        @test all(isfinite, interior(ay))
        @test all(isfinite, interior(az))
    end

    @testset "The nontraditional terms do no net work" begin
        # With the cell volumes V, Σ u V (f̃ʸ w)ᵤ − Σ w V (f̃ʸ u)_w = 0 and likewise for the v–w pair, provided the
        # normal velocities vanish on the boundaries. The nontraditional terms are the difference between the
        # nonhydrostatic and the hydrostatic accelerations.
        grid = rotated_grid(12)
        U = velocity_fields(grid, P)
        for (q, dim) in zip(U, 1:3)
            selectdim(interior(q), dim, [1, size(grid, dim) + 1]) .= 0
        end

        nonhydrostatic = coriolis_accelerations(grid, SphericalCoriolis(), U)
        hydrostatic = coriolis_accelerations(grid, HydrostaticSphericalCoriolis(), U)
        work = scale = 0.0
        for (q, a, aʰ) in zip(U, nonhydrostatic, hydrostatic)
            ℓ = map(ℓ -> ℓ(), location(q))
            V = compute!(Field(KernelFunctionOperation{location(q)...}(volume, grid, ℓ...)))
            δ = interior(q) .* interior(V) .* (interior(a) .- interior(aʰ))
            work += sum(δ)
            scale += sum(abs, δ)
        end

        @test scale > 0
        @test abs(work) < 1e-12 * scale
    end
end
