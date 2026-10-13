include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Coriolis: f̃ᶠᶜᵃ, f̃ʸᶠᶜᵃ, x_f̃_cross_U, y_f̃_cross_U, z_f̃_cross_U,
                            HydrostaticFormulation, NonhydrostaticFormulation,
                            TriadScheme, ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving, DualGridScheme
using Oceananigans.Grids: λnode, φnode
using Oceananigans.Operators: extrinsic_vector, volume
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid

schemes = (EnstrophyConserving(), EnergyConserving(), TriadScheme(),
           ActiveWeightedEnstrophyConserving(), ActiveWeightedEnergyConserving())

# We set the grid-frame components of a geographic velocity field on a rotated grid and compare the discrete
# Coriolis acceleration with the exact 2Ω × U, with a `LatitudeLongitudeGrid` as the reference for the error.

# A smooth geographic velocity field: eastward, northward, and upward components.
eastward_velocity(λ, φ)  = 20 * cosd(φ) + 5 * sind(2λ) * cosd(φ)
northward_velocity(λ, φ) = 10 * sind(λ) * cosd(φ)^2
upward_velocity(λ, φ)    = cosd(λ) * cosd(φ)

# Cartesian unit vectors pointing east and north at (λ, φ)
east_vector(λ, φ)  = [-sind(λ), cosd(λ), 0]
north_vector(λ, φ) = [-sind(φ) * cosd(λ), -sind(φ) * sind(λ), cosd(φ)]

# The rotation from the rotated frame of a grid with `north_pole = (λ₀, φ₀)` to the geographic frame,
# as used to build a `RotatedLatitudeLongitudeGrid`.
function frame_rotation(λ₀, φ₀)
    θ = 90 - φ₀
    Ry = [cosd(θ) 0 sind(θ); 0 1 0; -sind(θ) 0 cosd(θ)]
    Rz = [cosd(λ₀) -sind(λ₀) 0; sind(λ₀) cosd(λ₀) 0; 0 0 1]
    return Rz * Ry
end

function test_grids(arch, N; north_pole)
    kw = (size = (N, N, 2), halo = (3, 3, 3), z = (0, 1), longitude = (40, 80), latitude = (-20, 20))
    rotated_grid = RotatedLatitudeLongitudeGrid(arch; north_pole, topology = (Bounded, Bounded, Bounded), kw...)

    # Before rotation the rotated grid is this latitude-longitude grid, which therefore gives
    # each rotated-grid point's coordinates in the rotated frame.
    rotated_frame_grid = LatitudeLongitudeGrid(arch; kw...)

    return rotated_grid, rotated_frame_grid
end

# Exact geographic velocity and Coriolis acceleration (east, north, up) at (λ, φ).
function exact_velocity_and_coriolis(λ, φ, Ω)
    u = eastward_velocity(λ, φ)
    v = northward_velocity(λ, φ)
    w = upward_velocity(λ, φ)
    f = 2Ω * sind(φ)
    f̃ = 2Ω * cosd(φ)
    return (u, v, w), (f̃ * w - f * v, f * u, - f̃ * u)
end

# The exact Coriolis acceleration, or under the traditional approximation only f ẑ × U.
exact_coriolis(λ, φ, Ω, traditional::Bool) =
    traditional ? exact_traditional_coriolis(λ, φ, Ω) : last(exact_velocity_and_coriolis(λ, φ, Ω))

function exact_traditional_coriolis(λ, φ, Ω)
    (u, v, _), _ = exact_velocity_and_coriolis(λ, φ, Ω)
    f = 2Ω * sind(φ)
    return (- f * v, f * u, zero(f))
end

# Geographic coordinates and the grid's unit vectors x̂ and ŷ (as Cartesian vectors) at a point.
function point_geometry(i, j, ℓx, ℓy, grid, rotated_frame_grid, R)
    λ = λnode(i, j, 1, grid, ℓx, ℓy, Center())
    φ = φnode(i, j, 1, grid, ℓx, ℓy, Center())
    λ′ = λnode(i, j, 1, rotated_frame_grid, ℓx, ℓy, Center())
    φ′ = φnode(i, j, 1, rotated_frame_grid, ℓx, ℓy, Center())
    x̂ = R * east_vector(λ′, φ′)
    ŷ = R * north_vector(λ′, φ′)
    return λ, φ, x̂, ŷ
end

horizontal_vector(λ, φ, uᵉ, vⁿ) = uᵉ * east_vector(λ, φ) + vⁿ * north_vector(λ, φ)

# Set the grid-frame components of the geographic velocity, halos included.
function set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, R, Ω)
    for k in axes(u.data, 3), j in axes(u.data, 2), i in axes(u.data, 1)
        λ, φ, x̂, ŷ = point_geometry(i, j, Face(), Center(), grid, rotated_frame_grid, R)
        (uᵉ, vⁿ, _), _ = exact_velocity_and_coriolis(λ, φ, Ω)
        u.data[i, j, k] = horizontal_vector(λ, φ, uᵉ, vⁿ) ⋅ x̂
    end

    for k in axes(v.data, 3), j in axes(v.data, 2), i in axes(v.data, 1)
        λ, φ, x̂, ŷ = point_geometry(i, j, Center(), Face(), grid, rotated_frame_grid, R)
        (uᵉ, vⁿ, _), _ = exact_velocity_and_coriolis(λ, φ, Ω)
        v.data[i, j, k] = horizontal_vector(λ, φ, uᵉ, vⁿ) ⋅ ŷ
    end

    for k in axes(w.data, 3), j in axes(w.data, 2), i in axes(w.data, 1)
        λ = λnode(i, j, 1, grid, Center(), Center(), Center())
        φ = φnode(i, j, 1, grid, Center(), Center(), Center())
        (_, _, wᶻ), _ = exact_velocity_and_coriolis(λ, φ, Ω)
        w.data[i, j, k] = wᶻ
    end

    return nothing
end

function coriolis_accelerations(grid, coriolis, U)
    ax = Field(KernelFunctionOperation{Face, Center, Center}(x_f_cross_U, grid, coriolis, U))
    ay = Field(KernelFunctionOperation{Center, Face, Center}(y_f_cross_U, grid, coriolis, U))
    az = Field(KernelFunctionOperation{Center, Center, Face}(z_f_cross_U, grid, coriolis, U))
    compute!(ax)
    compute!(ay)
    compute!(az)
    return ax, ay, az
end

"""
Return the maximum errors of the discrete Coriolis acceleration on `grid`, relative to the
largest exact acceleration: in the grid-frame components at the velocity points, and in the
geographic components at cell centers.
"""
function coriolis_errors(grid, rotated_frame_grid, coriolis; north_pole, traditional = false)
    R = frame_rotation(north_pole...)
    Ω = coriolis.rotation_rate

    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZFaceField(grid)
    set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, R, Ω)

    ax, ay, az = coriolis_accelerations(grid, coriolis, (u, v, w))

    exact(λ, φ) = exact_coriolis(λ, φ, Ω, traditional)

    Nx, Ny, Nz = size(grid)
    grid_frame_error = geographic_error = scale = 0.0

    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        # Grid-frame components at the velocity points
        λ, φ, x̂, _ = point_geometry(i, j, Face(), Center(), grid, rotated_frame_grid, R)
        aᵉ, aⁿ, _ = exact(λ, φ)
        grid_frame_error = max(grid_frame_error, abs(ax[i, j, k] - horizontal_vector(λ, φ, aᵉ, aⁿ) ⋅ x̂))

        λ, φ, _, ŷ = point_geometry(i, j, Center(), Face(), grid, rotated_frame_grid, R)
        aᵉ, aⁿ, _ = exact(λ, φ)
        grid_frame_error = max(grid_frame_error, abs(ay[i, j, k] - horizontal_vector(λ, φ, aᵉ, aⁿ) ⋅ ŷ))

        λ = λnode(i, j, k, grid, Center(), Center(), Face())
        φ = φnode(i, j, k, grid, Center(), Center(), Face())
        aᵉ, aⁿ, aᶻ = exact(λ, φ)
        grid_frame_error = max(grid_frame_error, abs(az[i, j, k] - aᶻ))

        # Geographic components at cell centers
        λ = λnode(i, j, k, grid, Center(), Center(), Center())
        φ = φnode(i, j, k, grid, Center(), Center(), Center())
        aᵉ, aⁿ, _ = exact(λ, φ)
        axᶜᶜᶜ = (ax[i, j, k] + ax[i+1, j, k]) / 2
        ayᶜᶜᶜ = (ay[i, j, k] + ay[i, j+1, k]) / 2
        a₁, a₂ = extrinsic_vector(i, j, k, grid, axᶜᶜᶜ, ayᶜᶜᶜ)
        geographic_error = max(geographic_error, abs(a₁ - aᵉ), abs(a₂ - aⁿ))

        scale = max(scale, abs(aᵉ), abs(aⁿ))
    end

    return grid_frame_error / scale, geographic_error / scale
end

@testset "SphericalCoriolis on rotated grids" begin
    # The comparison sets and reads fields one point at a time, so it runs on the CPU.
    let arch = CPU()
        Oceananigans.defaults.FloatType = Float64

        resolutions = (8, 16)

        @testset "Rotated grid matches the latitude-longitude grid to discretization error" begin
            # Every scheme, with the nontraditional terms (NonhydrostaticFormulation) and without them.
            # The active-weighted schemes are left out: their wet-point correction counts the wall faces of
            # this bounded grid as inactive, which is only consistent with a velocity that vanishes there, and
            # the analytic field does not.
            cases = [(summary(scheme) * " " * summary(formulation),
                      SphericalCoriolis(; scheme, formulation),
                      formulation isa HydrostaticFormulation)
                     for (scheme, formulation) in ((EnstrophyConserving(), NonhydrostaticFormulation()),
                                                   (TriadScheme(), NonhydrostaticFormulation()),
                                                   (EnergyConserving(), HydrostaticFormulation()))]

            for (name, coriolis, traditional) in cases
                @testset "$name" begin
                    rotated_errors = map(resolutions) do N
                        grid, rotated_frame_grid = test_grids(arch, N; north_pole = (0, 0))
                        coriolis_errors(grid, rotated_frame_grid, coriolis; north_pole = (0, 0), traditional)
                    end

                    reference_errors = map(resolutions) do N
                        _, latitude_longitude_grid = test_grids(arch, N; north_pole = (0, 0))
                        coriolis_errors(latitude_longitude_grid, latitude_longitude_grid, coriolis;
                                        north_pole = (0, 90), traditional)
                    end

                    for n in 1:length(resolutions)
                        # Discretization errors are small, and on the rotated grid comparable
                        # to those on the latitude-longitude grid
                        @test all(<(1e-2), rotated_errors[n])
                        @test all(<(1e-2), reference_errors[n])
                        @test all(rotated_errors[n] .< 4 .* reference_errors[n])
                    end

                    # Second-order convergence
                    for n in 2:length(resolutions)
                        @test all(rotated_errors[n-1] ./ rotated_errors[n] .> 3)
                    end
                end
            end
        end

        @testset "HydrostaticSphericalCoriolis has no vertical component on a rotated grid" begin
            grid, rotated_frame_grid = test_grids(arch, 8; north_pole = (0, 0))
            u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
            coriolis = HydrostaticSphericalCoriolis()
            set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, frame_rotation(0, 0), coriolis.rotation_rate)
            _, _, az = coriolis_accelerations(grid, coriolis, (u, v, w))
            @test all(iszero, interior(az))
        end

        @testset "The nonhydrostatic formulation reduces to the hydrostatic one without the nontraditional terms" begin
            # The nontraditional terms multiply w in the horizontal components and u, v in the vertical one,
            # so with w ≡ 0 every scheme's horizontal acceleration is its traditional stencil exactly, and with
            # u ≡ v ≡ 0 the vertical acceleration vanishes.
            grid, rotated_frame_grid = test_grids(arch, 8; north_pole = (0, 0))
            R = frame_rotation(0, 0)

            for scheme in schemes
                nonhydrostatic = SphericalCoriolis(; scheme)
                hydrostatic = SphericalCoriolis(; scheme, formulation = HydrostaticFormulation())
                Ω = nonhydrostatic.rotation_rate

                u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
                set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, R, Ω)

                fill!(w, 0)
                ax, ay, _ = coriolis_accelerations(grid, nonhydrostatic, (u, v, w))
                ax_hydrostatic, ay_hydrostatic, _ = coriolis_accelerations(grid, hydrostatic, (u, v, w))
                @test interior(ax) == interior(ax_hydrostatic)
                @test interior(ay) == interior(ay_hydrostatic)

                set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, R, Ω)
                fill!(u, 0)
                fill!(v, 0)
                _, _, az = coriolis_accelerations(grid, nonhydrostatic, (u, v, w))
                @test all(iszero, interior(az))
            end
        end

        @testset "The nontraditional terms do no net work" begin
            # With the cell volumes V = Az Δz, Σ u V (f̃ʸ w)ᵤ − Σ w V (f̃ʸ u)_w = 0 and likewise for the v–w pair,
            # provided the normal velocities vanish on the boundaries so that every product is counted on both
            # sides. The wall and top/bottom values of the analytic field are therefore set to zero.
            grid, rotated_frame_grid = test_grids(arch, 12; north_pole = (0, 0))
            Nx, Ny, Nz = size(grid)
            coriolis = SphericalCoriolis()
            u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
            set_geographic_velocity!(u, v, w, grid, rotated_frame_grid, frame_rotation(0, 0), coriolis.rotation_rate)
            interior(u)[1, :, :] .= 0; interior(u)[Nx+1, :, :] .= 0
            interior(v)[:, 1, :] .= 0; interior(v)[:, Ny+1, :] .= 0
            interior(w)[:, :, 1] .= 0; interior(w)[:, :, Nz+1] .= 0
            U = (u, v, w)

            work = scale = 0.0
            for k in 1:Nz, j in 1:Ny, i in 1:Nx+1
                δ = u[i, j, k] * volume(i, j, k, grid, Face(), Center(), Center()) * x_f̃_cross_U(i, j, k, grid, coriolis, U)
                work += δ; scale += abs(δ)
            end
            for k in 1:Nz, j in 1:Ny+1, i in 1:Nx
                δ = v[i, j, k] * volume(i, j, k, grid, Center(), Face(), Center()) * y_f̃_cross_U(i, j, k, grid, coriolis, U)
                work += δ; scale += abs(δ)
            end
            for k in 1:Nz+1, j in 1:Ny, i in 1:Nx
                δ = w[i, j, k] * volume(i, j, k, grid, Center(), Center(), Face()) * z_f̃_cross_U(i, j, k, grid, coriolis, U)
                work += δ; scale += abs(δ)
            end

            @test scale > 0
            @test abs(work) < 1e-12 * scale
        end

        @testset "Unsupported schemes are rejected rather than dropping the nontraditional terms" begin
            grid, _ = test_grids(arch, 8; north_pole = (0, 0))
            @test_throws ArgumentError SphericalCoriolis(scheme = DualGridScheme(grid))
            @test HydrostaticSphericalCoriolis(scheme = DualGridScheme(grid)) isa SphericalCoriolis
        end

        @testset "Unrotated grids" begin
            # A rotated grid whose pole is the geographic pole is a latitude-longitude grid
            N = 8
            _, latitude_longitude_grid = test_grids(arch, N; north_pole = (0, 0))
            unrotated_grid, _ = test_grids(arch, N; north_pole = (0, 90))

            coriolis = SphericalCoriolis()
            R = frame_rotation(0, 90)
            Ω = coriolis.rotation_rate

            accelerations = map((latitude_longitude_grid, unrotated_grid)) do grid
                u, v, w = XFaceField(grid), YFaceField(grid), ZFaceField(grid)
                set_geographic_velocity!(u, v, w, grid, latitude_longitude_grid, R, Ω)
                coriolis_accelerations(grid, coriolis, (u, v, w))
            end

            for (a_latitude_longitude, a_unrotated) in zip(accelerations...)
                @test interior(a_unrotated) ≈ interior(a_latitude_longitude) rtol=1e-10
            end

            # On the latitude-longitude grid the horizontal rotation vector is f̃ along y
            @test all(f̃ʸᶠᶜᵃ(i, j, 1, latitude_longitude_grid, coriolis) === f̃ᶠᶜᵃ(i, j, 1, latitude_longitude_grid, coriolis)
                      for j in 1:N, i in 1:N+1)
        end
    end
end
