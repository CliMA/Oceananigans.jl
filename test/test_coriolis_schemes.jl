include("dependencies_for_runtests.jl")

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving
using Oceananigans.Coriolis: fᶜᶜᵃ, fᶠᶠᵃ, HydrostaticFormulation
using Oceananigans.Coriolis: 𝒯⁺⁺, 𝒯⁻⁺, 𝒯⁺⁻, 𝒯⁻⁻

#####
##### Helpers
#####

function make_velocity_fields(grid, FT; u_val=FT(0), v_val=FT(0))
    u = Field{Face, Center, Center}(grid)
    v = Field{Center, Face, Center}(grid)
    w = Field{Center, Center, Face}(grid)
    fill!(u, u_val)
    fill!(v, v_val)
    fill!(w, FT(0))
    fill_halo_regions!((u, v, w))
    return (u=u, v=v, w=w)
end

#####
##### 1. Instantiation tests for new scheme types
#####

@testset "Coriolis scheme instantiation" begin
    @info "Testing Coriolis scheme instantiation..."

    for FT in float_types
        coriolis = SphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())
        @test coriolis.scheme isa ActiveWeightedEnstrophyConserving

        coriolis = SphericalCoriolis(FT, scheme=ActiveWeightedEnergyConserving())
        @test coriolis.scheme isa ActiveWeightedEnergyConserving

        coriolis = SphericalCoriolis(FT, scheme=EENConserving())
        @test coriolis.scheme isa EENConserving

        coriolis = HydrostaticSphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())
        @test coriolis.scheme isa ActiveWeightedEnstrophyConserving

        coriolis = HydrostaticSphericalCoriolis(FT, scheme=EENConserving())
        @test coriolis.scheme isa EENConserving

        # Default scheme for HydrostaticSphericalCoriolis is EENConserving
        coriolis = HydrostaticSphericalCoriolis(FT)
        @test coriolis.scheme isa EENConserving
    end
end

#####
##### 2. Stencil correctness: uniform velocity on LatLonGrid
#####
##### On a regular-in-longitude LatLonGrid, fᶜᶜᵃ depends only on j.
##### For uniform v, ℑxᶠᵃᵃ(fᶜᶜᵃ) = fᶜᶜᵃ and ℑxyᶠᶜᵃ(v) = v,
##### so EnstrophyConserving gives exactly -fᶜᶜᵃ(i,j) * v.
#####

function test_enstrophy_conserving_uniform_v(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    U = make_velocity_fields(grid, FT; v_val=FT(1))

    i, j, k = 4, 4, 1
    result = x_f_cross_U(i, j, k, grid, coriolis, U)
    expected = -fᶜᶜᵃ(i, j, k, grid, coriolis)
    @test result ≈ expected
end

function test_enstrophy_conserving_uniform_u(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))
    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    U = make_velocity_fields(grid, FT; u_val=FT(1))

    i, j, k = 4, 4, 1
    result = y_f_cross_U(i, j, k, grid, coriolis, U)
    # ℑyᵃᶠᵃ(fᶜᶜᵃ) averages f at j-1 and j (two center latitudes → face latitude)
    expected = FT(0.5) * (fᶜᶜᵃ(i, j-1, k, grid, coriolis) + fᶜᶜᵃ(i, j, k, grid, coriolis))
    @test result ≈ expected
end

#####
##### 3. Active-weighted = plain on flat bottom 
#####
##### On a flat-bottom grid, all neighboring velocity nodes are active,
#####

function test_active_weighted_equals_plain_flat_bottom(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))

    cor_ens    = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    cor_aw_ens = HydrostaticSphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())
    cor_ene    = HydrostaticSphericalCoriolis(FT, scheme=EnergyConserving())
    cor_aw_ene = HydrostaticSphericalCoriolis(FT, scheme=ActiveWeightedEnergyConserving())

    U = make_velocity_fields(grid, FT; v_val=FT(1), u_val=FT(0.5))

    i, j, k = 4, 4, 1

    # Enstrophy: plain vs active-weighted
    @test x_f_cross_U(i, j, k, grid, cor_aw_ens, U) ≈ x_f_cross_U(i, j, k, grid, cor_ens, U)
    @test y_f_cross_U(i, j, k, grid, cor_aw_ens, U) ≈ y_f_cross_U(i, j, k, grid, cor_ens, U)

    # Energy: plain vs active-weighted
    @test x_f_cross_U(i, j, k, grid, cor_aw_ene, U) ≈ x_f_cross_U(i, j, k, grid, cor_ene, U)
    @test y_f_cross_U(i, j, k, grid, cor_aw_ene, U) ≈ y_f_cross_U(i, j, k, grid, cor_ene, U)
end

#####
##### 4. EEN triad structure verification
#####
##### Each triad at T-point (i,j) sums 3 of the 4 surrounding f-points,
##### omitting the corner indicated by the superscript sign pattern.
##### On a LatLonGrid, fᶠᶠᵃ depends on j only, so f(i,j) = f(i+1,j).
#####
##### Mapping: 𝒯⁺⁺ omits SW, 𝒯⁻⁺ omits SE, 𝒯⁺⁻ omits NW, 𝒯⁻⁻ omits NE
#####

function test_een_triad_structure(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (4, 4, 1),
                                 latitude = (44, 48),
                                 longitude = (0, 4),
                                 z = (0, 1))

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EENConserving())

    i, j, k = 2, 2, 1

    # The 4 f-points surrounding T(i,j)
    f_sw = fᶠᶠᵃ(i,   j,   k, grid, coriolis)
    f_se = fᶠᶠᵃ(i+1, j,   k, grid, coriolis)
    f_nw = fᶠᶠᵃ(i,   j+1, k, grid, coriolis)
    f_ne = fᶠᶠᵃ(i+1, j+1, k, grid, coriolis)

    # On LatLonGrid, f is i-independent
    @test f_sw ≈ f_se
    @test f_nw ≈ f_ne

    # Verify triad definitions: each omits one corner
    # 𝒯⁺⁺ omits SW → sums NW + NE + SE
    @test 𝒯⁺⁺(i, j, k, grid, coriolis) ≈ f_nw + f_ne + f_se
    # 𝒯⁻⁺ omits SE → sums SW + NW + NE
    @test 𝒯⁻⁺(i, j, k, grid, coriolis) ≈ f_sw + f_nw + f_ne
    # 𝒯⁺⁻ omits NW → sums NE + SE + SW
    @test 𝒯⁺⁻(i, j, k, grid, coriolis) ≈ f_ne + f_se + f_sw
    # 𝒯⁻⁻ omits NE → sums SE + SW + NW
    @test 𝒯⁻⁻(i, j, k, grid, coriolis) ≈ f_se + f_sw + f_nw

    # For constant f, each triad = 3f. Near the pole f is approximately constant.
    grid_pole = LatitudeLongitudeGrid(CPU(), FT,
                                      size = (4, 4, 1),
                                      latitude = (89, 90),
                                      longitude = (0, 4),
                                      z = (0, 1))

    f_mean = fᶠᶠᵃ(2, 2, 1, grid_pole, coriolis)
    for triad_fn in (𝒯⁺⁺, 𝒯⁻⁺, 𝒯⁺⁻, 𝒯⁻⁻)
        @test abs(triad_fn(2, 2, 1, grid_pole, coriolis) - 3 * f_mean) / abs(3 * f_mean) < 0.01
    end
end

#####
##### 5. Antisymmetry: x_f_cross_U ∝ -fv, y_f_cross_U ∝ +fu
#####

function test_coriolis_antisymmetry(FT, scheme)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))
    coriolis = HydrostaticSphericalCoriolis(FT, scheme=scheme)

    i, j, k = 4, 4, 1

    U_v = make_velocity_fields(grid, FT; v_val=FT(1))
    U_u = make_velocity_fields(grid, FT; u_val=FT(1))

    fx = x_f_cross_U(i, j, k, grid, coriolis, U_v)
    fy = y_f_cross_U(i, j, k, grid, coriolis, U_u)

    # Northern hemisphere: f > 0, so x-tendency = -fv < 0, y-tendency = +fu > 0
    @test fx < 0
    @test fy > 0
end

@testset "Coriolis scheme stencil correctness" begin
    @info "Testing Coriolis scheme stencil correctness..."

    for FT in float_types
        @testset "EnstrophyConserving uniform velocity [$FT]" begin
            test_enstrophy_conserving_uniform_v(FT)
            test_enstrophy_conserving_uniform_u(FT)
        end

        @testset "ActiveWeighted = plain on flat bottom [$FT]" begin
            test_active_weighted_equals_plain_flat_bottom(FT)
        end

        @testset "EEN triad structure [$FT]" begin
            test_een_triad_structure(FT)
        end

        @testset "Antisymmetry [$FT]" begin
            for scheme in (EnstrophyConserving(),
                           EnergyConserving(),
                           ActiveWeightedEnstrophyConserving(),
                           ActiveWeightedEnergyConserving(),
                           EENConserving())
                test_coriolis_antisymmetry(FT, scheme)
            end
        end
    end
end

#####
##### 6. Immersed boundary: Jamart wet-point correction
#####
##### We test that:
##### (a) Active-weighted = plain in the ocean interior (far from boundaries)
##### (b) Active-weighted compensates for masked nodes near the coast
#####

using Oceananigans.ImmersedBoundaries: GridFittedBottom

function test_jamart_correction_near_topography(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (40, 48),
                                 longitude = (0, 8),
                                 z = (-1, 0))

    # Land for φ < 44°, ocean for φ ≥ 44° (j ≤ 4 is land)
    bottom(λ, φ) = φ < 44 ? 0.0 : -1.0
    ib_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

    cor_plain = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    cor_aw    = HydrostaticSphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())

    U = make_velocity_fields(ib_grid, FT; v_val=FT(1))

    # (a) Deep interior: all neighbors active → schemes agree
    i_ocean, j_ocean, k = 4, 7, 1
    @test x_f_cross_U(i_ocean, j_ocean, k, ib_grid, cor_aw, U) ≈
          x_f_cross_U(i_ocean, j_ocean, k, ib_grid, cor_plain, U)

    # (b) Near coast: some v-neighbors are masked → active-weighted compensates
    i_coast, j_coast = 4, 5
    result_aw    = x_f_cross_U(i_coast, j_coast, k, ib_grid, cor_aw, U)
    result_plain = x_f_cross_U(i_coast, j_coast, k, ib_grid, cor_plain, U)

    # Both should be negative (f > 0, v > 0)
    @test result_aw < 0
    @test result_plain < 0

    # Active-weighted has larger magnitude: it divides by active_nodes < 1
    # to compensate for the missing (masked) neighbors
    @test abs(result_aw) >= abs(result_plain) - eps(FT)
end

@testset "Immersed boundary Coriolis (Jamart correction)" begin
    @info "Testing Coriolis Jamart correction near topography..."
    for FT in float_types
        test_jamart_correction_near_topography(FT)
    end
end

#####
##### 7. Geostrophic balance test (NEMO CANAL style)
#####
##### A zonal geostrophic jet u(y) with SSH η = -f * u * y/g should remain
##### steady under Coriolis forcing (pressure gradient balances Coriolis).
##### We test that the Coriolis tendency -fv ≈ 0 when v = 0 in geostrophic balance.
#####

function test_geostrophic_balance_steady(FT, scheme)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (8, 8, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (-100, 0),
                                 topology = (Periodic, Bounded, Bounded))

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=scheme)

    model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                          momentum_advection = nothing,
                                          buoyancy = nothing,
                                          tracers = nothing,
                                          closure = nothing)

    # Zonal geostrophic jet: u = U₀, v = 0
    # The free surface adjusts to balance Coriolis
    U₀ = FT(0.1)
    set!(model, u=U₀)

    Ω = coriolis.rotation_rate
    f_mid = 2Ω * sind(FT(45))
    T_inertial = 2π / f_mid
    Δt = T_inertial / 200

    simulation = Simulation(model, Δt=Δt, stop_time=10Δt)
    run!(simulation)

    # v should remain small (not excited by Coriolis alone without pressure imbalance)
    v_max = maximum(abs, interior(model.velocities.v))
    @test v_max < U₀ # v should not grow to the scale of u
end

@testset "Geostrophic balance" begin
    @info "Testing geostrophic balance..."

    for scheme in (EnstrophyConserving(),
                   EnergyConserving(),
                   ActiveWeightedEnstrophyConserving(),
                   EENConserving())
        @testset "scheme=$(summary(scheme))" begin
            test_geostrophic_balance_steady(Float64, scheme)
        end
    end
end

#####
##### 8. Energy conservation under Coriolis
#####
##### Initialize with a Gaussian anticyclonic eddy.
##### Coriolis is a rotation — it should not change total kinetic energy.
##### Run for a few time steps and check KE is conserved.
#####

function test_coriolis_energy_conservation(FT, scheme)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (16, 16, 1),
                                 latitude = (30, 60),
                                 longitude = (0, 30),
                                 z = (-100, 0),
                                 topology = (Periodic, Bounded, Bounded))

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=scheme)

    model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                          momentum_advection = nothing,
                                          buoyancy = nothing,
                                          tracers = nothing,
                                          closure = nothing)

    # Gaussian anticyclonic eddy initial condition (simplified NEMO VORTEX)
    λ₀, φ₀ = FT(15), FT(45)  # eddy center
    σ = FT(5)                  # eddy width in degrees
    u₀(λ, φ, z) = 0.1 * (φ - φ₀) / σ * exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / (2σ^2))
    v₀(λ, φ, z) = -0.1 * (λ - λ₀) / σ * exp(-((λ - λ₀)^2 + (φ - φ₀)^2) / (2σ^2))
    set!(model, u=u₀, v=v₀)

    u, v, w = model.velocities
    KE_op = @at (Center, Center, Center) (u^2 + v^2) / 2
    KE_field = Field(KE_op)
    compute!(KE_field)
    KE_initial = sum(KE_field)

    Ω = coriolis.rotation_rate
    f_mid = 2Ω * sind(FT(45))
    T_inertial = 2π / f_mid
    Δt = T_inertial / 100

    simulation = Simulation(model, Δt=Δt, stop_time=5Δt)
    run!(simulation)

    compute!(KE_field)
    KE_final = sum(KE_field)

    # Coriolis should not inject or remove energy
    @test abs(KE_final - KE_initial) / abs(KE_initial) < 0.05
end

@testset "Energy conservation (NEMO VORTEX style)" begin
    @info "Testing energy conservation under Coriolis..."

    for scheme in (EnstrophyConserving(),
                   EnergyConserving(),
                   ActiveWeightedEnstrophyConserving(),
                   EENConserving())
        @testset "scheme=$(summary(scheme))" begin
            test_coriolis_energy_conservation(Float64, scheme)
        end
    end
end

#####
##### 9. Inertial oscillation on doubly-periodic f-plane
#####
##### Uniform initial u=u₀, v=0 on a doubly-periodic RectilinearGrid with FPlane.
##### After one inertial period T=2π/f, velocities should return to initial values.
##### Uses doubly-periodic to avoid boundary effects on v-points.
#####

function test_inertial_oscillation(FT, scheme)
    grid = RectilinearGrid(CPU(), FT,
                           size = (4, 4, 1),
                           x = (0, 1e5),
                           y = (0, 1e5),
                           z = (-100, 0),
                           topology = (Periodic, Periodic, Bounded))

    f₀ = FT(1e-4)
    coriolis = FPlane(FT, f=f₀, scheme=scheme)

    model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                        momentum_advection = nothing,
                                        buoyancy = nothing,
                                        tracers = nothing,
                                        closure = nothing)

    u₀ = FT(0.1)
    set!(model, u=u₀)

    T_inertial = 2π / f₀
    Δt = T_inertial / 400
    simulation = Simulation(model, Δt=Δt, stop_time=T_inertial)
    run!(simulation)

    u_final = model.velocities.u[2, 2, 1]
    v_final = model.velocities.v[2, 2, 1]

    @test abs(u_final - u₀) / u₀ < 0.05
    @test abs(v_final) / u₀ < 0.05
end

@testset "Inertial oscillation (f-plane)" begin
    @info "Testing inertial oscillation on f-plane..."

    for scheme in (EnstrophyConserving(),
                   EnergyConserving(),
                   ActiveWeightedEnstrophyConserving(),
                   ActiveWeightedEnergyConserving(),
                   EENConserving())
        @testset "scheme=$(summary(scheme))" begin
            test_inertial_oscillation(Float64, scheme)
        end
    end
end
