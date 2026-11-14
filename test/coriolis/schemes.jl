include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: EnergyConserving, EnstrophyConserving, horizontal_advection_U, horizontal_advection_V
using Oceananigans.Coriolis: fᶜᶜᵃ, fᶠᶠᵃ, HydrostaticFormulation, TriadScheme, ActiveWeightedEnergyConserving, ActiveWeightedEnstrophyConserving, DualGridScheme
using Oceananigans.Coriolis: 𝒯⁺⁺, 𝒯⁻⁺, 𝒯⁺⁻, 𝒯⁻⁻
using Oceananigans.Operators: Ayᶜᶠᶜ, Ayᶠᶜᶜ, Axᶠᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ, Vᶠᶜᶜ, Vᶜᶠᶜ, Azᶜᶜᶜ
using Oceananigans.Grids: peripheral_node
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Random: Random

#####
##### Helpers
#####

function make_velocity_fields(grid; u_val=0, v_val=0)
    u = Field{Face, Center, Center}(grid)
    v = Field{Center, Face, Center}(grid)
    w = Field{Center, Center, Face}(grid)
    fill!(u, u_val)
    fill!(v, v_val)
    fill!(w, 0)
    fill_halo_regions!((u, v, w))
    return (u=u, v=v, w=w)
end

#####
##### 2. Stencil correctness: uniform velocity on LatLonGrid
#####
##### On a regular-in-longitude LatLonGrid, fᶠᶠᵃ depends only on j.
##### For uniform v, the EnstrophyConserving x-tendency uses
##### ℑyᵃᶜᵃ(fᶠᶠᵃ) = (fᶠᶠᵃ(j) + fᶠᶠᵃ(j+1))/2 as the f-factor.
##### Area-weighted interpolation of v is approximately v on a LatLonGrid.
#####

function test_enstrophy_conserving_uniform_v(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (6, 6, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    U = make_velocity_fields(grid; v_val=1)

    i, j, k = 3, 3, 1
    result = x_f_cross_U(i, j, k, grid, coriolis, U)
    # f-factor: ℑyᵃᶜᵃ(fᶠᶠᵃ) averages f at the two face latitudes bracketing center j
    f_interp = FT(0.5) * (fᶠᶠᵃ(i, j, k, grid, coriolis) + fᶠᶠᵃ(i, j+1, k, grid, coriolis))
    # area ratio: ℑxyᶠᶜᵃ(Ayᶜᶠ) / Ayᶠᶜ accounts for Ay varying with latitude
    area_ratio = FT(0.5) * (Ayᶜᶠᶜ(i, j, k, grid) + Ayᶜᶠᶜ(i, j+1, k, grid)) / Ayᶠᶜᶜ(i, j, k, grid)
    expected = -f_interp * area_ratio
    @test result ≈ expected
end

function test_enstrophy_conserving_uniform_u(FT)
    grid = LatitudeLongitudeGrid(CPU(), FT,
                                 size = (6, 6, 1),
                                 latitude = (44, 46),
                                 longitude = (0, 8),
                                 z = (0, 1))
    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnstrophyConserving())
    U = make_velocity_fields(grid; u_val=1)

    i, j, k = 3, 3, 1
    result = y_f_cross_U(i, j, k, grid, coriolis, U)
    # ℑxᶜᵃᵃ(fᶠᶠᵃ) is i-independent on a LatLonGrid, so equals fᶠᶠᵃ at face latitude j
    expected = fᶠᶠᵃ(i, j, k, grid, coriolis)
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

    U = make_velocity_fields(grid; v_val=1, u_val=0.5)

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

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=TriadScheme())

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

    U_v = make_velocity_fields(grid; v_val=1)
    U_u = make_velocity_fields(grid; u_val=1)

    fx = x_f_cross_U(i, j, k, grid, coriolis, U_v)
    fy = y_f_cross_U(i, j, k, grid, coriolis, U_u)

    # Northern hemisphere: f > 0, so x-tendency = -fv < 0, y-tendency = +fu > 0
    @test fx < 0
    @test fy > 0
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

    U = make_velocity_fields(ib_grid; v_val=1)

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

#####
##### 7. Geostrophic balance test (NEMO CANAL style)
#####
##### A zonal geostrophic jet u(y) with SSH η = -f * u * y/g should remain
##### steady under Coriolis forcing (pressure gradient balances Coriolis).
##### We test that the Coriolis tendency -fv ≈ 0 when v = 0 in geostrophic balance.
#####

function test_geostrophic_balance_steady(FT, arch, scheme)
    grid = LatitudeLongitudeGrid(arch, FT,
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
    T = 2π / (2Ω * sind(FT(45)))
    Δt = T / 200

    simulation = Simulation(model; Δt, stop_time=10Δt, verbose=false)
    run!(simulation)

    # v should remain small (not excited by Coriolis alone without pressure imbalance)
    vmax = maximum(abs, interior(model.velocities.v))
    @test vmax < U₀ # v should not grow to the scale of u
end

#####
##### 8. Energy conservation under Coriolis
#####
##### Initialize with a Gaussian anticyclonic eddy.
##### Coriolis is a rotation — it should not change total kinetic energy.
##### Run for a few time steps and check KE is within bounds.
#####

function test_coriolis_energy_conservation(FT, arch, scheme)
    grid = LatitudeLongitudeGrid(arch, FT,
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
    KE = Field(KE_op)
    compute!(KE)
    KEᵢ = sum(KE)

    Ω = coriolis.rotation_rate
    T = 2π / (2Ω * sind(FT(45)))
    Δt = T / 100

    simulation = Simulation(model; Δt, stop_time=5Δt, verbose=false)
    run!(simulation)

    compute!(KE)
    KEₑ = sum(KE)

    # Coriolis should not inject or remove energy
    @test abs(KEₑ - KEᵢ) / abs(KEᵢ) < 0.003
end

#####
##### 9. Inertial oscillation on doubly-periodic f-plane
#####
##### Uniform initial u=u₀, v=0 on a doubly-periodic RectilinearGrid with FPlane.
##### After one inertial period T=2π/f, velocities should return to initial values.
##### Uses doubly-periodic to avoid boundary effects on v-points.
#####

function test_inertial_oscillation(FT, arch, scheme)
    grid = RectilinearGrid(arch, FT,
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

    T = 2π / f₀
    Δt = T / 400
    simulation = Simulation(model; Δt, stop_time=T, verbose=false)
    run!(simulation)

    CUDA.@allowscalar uₑ = model.velocities.u[2, 2, 1]
    CUDA.@allowscalar vₑ = model.velocities.v[2, 2, 1]

    @test abs(uₑ - u₀) / u₀ < 0.01
    @test abs(vₑ) / u₀ < 0.01
end

#####
##### C-D scheme: the 2Δx meridional velocity, invisible to the averaged C-grid Coriolis, oscillates inertially
#####

function test_dual_grid_scheme_inertial_oscillations(FT, arch, timestepper)
    grid = RectilinearGrid(arch, FT, size=(8, 4, 1), x=(0, 8e4), y=(0, 4e4), z=(-100, 0), topology=(Periodic, Periodic, Bounded))

    f₀ = FT(1e-4)
    T = 2π / f₀
    Δt = T / 400

    model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(FT, f=f₀, scheme=DualGridScheme(grid)), timestepper,
                                        momentum_advection=nothing, buoyancy=nothing, tracers=nothing, closure=nothing)

    v₀ = FT(0.1)
    set!(model, v=[isodd(i) ? v₀ : -v₀ for i in 1:8, j in 1:4, k in 1:1])
    run!(Simulation(model; Δt, stop_time=T/4, verbose=false))
    @test maximum(abs, interior(model.velocities.v)) / v₀ < 0.01

    model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(FT, f=f₀, scheme=DualGridScheme(grid)), timestepper,
                                        momentum_advection=nothing, buoyancy=nothing, tracers=nothing, closure=nothing)

    u₀ = FT(0.1)
    set!(model, u=u₀)
    run!(Simulation(model; Δt, stop_time=T/4, verbose=false))
    @test maximum(abs, interior(model.velocities.u)) / u₀ < 0.01
    @test all(v -> abs(v + u₀) / u₀ < 0.01, Array(interior(model.velocities.v)))
end

function test_dual_grid_scheme_checkpoint_restart(FT, arch, timestepper)
    grid = RectilinearGrid(arch, FT, size=(8, 4, 1), x=(0, 8e4), y=(0, 4e4), z=(-100, 0), topology=(Periodic, Periodic, Bounded))
    dir = mktempdir()
    prefix = "dual_grid_scheme_$(timestepper)_$FT"
    checkpoint_iteration = 5
    stop_iteration = 2checkpoint_iteration

    function dual_grid_simulation()
        model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(FT, f=1e-4, scheme=DualGridScheme(grid)), timestepper,
                                            momentum_advection=nothing, buoyancy=nothing, tracers=nothing, closure=nothing)
        set!(model, u=(x, y, z) -> sin(2π * x / 8e4) / 10, v=(x, y, z) -> cos(2π * y / 4e4) / 10)
        simulation = Simulation(model; Δt=100, stop_iteration, verbose=false)
        simulation.output_writers[:checkpointer] = Checkpointer(model; schedule=IterationInterval(checkpoint_iteration), dir, prefix, cleanup=false)
        return simulation
    end

    uninterrupted = dual_grid_simulation()
    run!(uninterrupted)

    restarted = dual_grid_simulation()
    checkpoint_filepath = joinpath(dir, "$(prefix)_iteration$(checkpoint_iteration).jld2")
    pickup_log = (:info, Regex("^Picking up simulation from checkpoint file \\Q$(checkpoint_filepath)\\E; last modified \\(UTC\\): "))
    @test_logs pickup_log run!(restarted, pickup=checkpoint_iteration)

    @test Array(interior(restarted.model.velocities.u)) == Array(interior(uninterrupted.model.velocities.u))
    @test Array(interior(restarted.model.velocities.v)) == Array(interior(uninterrupted.model.velocities.v))

    rm(dir; recursive=true)
end

#####
##### 10. Coriolis work with horizontally varying Δz (PartialCellBottom)
#####
##### The discrete Coriolis work Σ Δx Aˣ u (f × u)ˣ + Σ Δy Aʸ v (f × u)ʸ should vanish for any u, v,
##### and so should the work of the VectorInvariant vorticity term (ζ × u).
##### With a single level over a PartialCellBottom, Δz varies by orders of magnitude between neighbors.
##### The stretched latitude grid has Az ≠ Δx Δy, so the weights Δx Aˣ and V differ there.
#####

function partial_cell_grid(underlying_grid; seed=1234)
    Nx, Ny, _ = size(underlying_grid)
    FT = eltype(underlying_grid)
    H = FT(4000)
    Random.seed!(seed)
    depths = H .* (FT(0.002) .+ FT(0.998) .* rand(FT, Nx, Ny))
    depths[1:2, 1:2] .= 0 # some land
    return ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(-depths; minimum_fractional_cell_height=FT(1e-3)))
end

function random_masked_velocities(grid; seed=42)
    Random.seed!(seed)
    FT = eltype(grid)
    U = make_velocity_fields(grid)
    for (velocity, location) in ((U.u, (Face(), Center(), Center())), (U.v, (Center(), Face(), Center())))
        Nx, Ny, Nz = size(velocity)
        data = [peripheral_node(i, j, k, grid, location...) ? zero(FT) : randn(FT) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        set!(velocity, data)
    end
    fill_halo_regions!((U.u, U.v))
    return U
end

function relative_work(grid, U, Fu, Fv)
    Nx, Ny, _ = size(grid)
    work = zero(eltype(grid))
    scale = zero(eltype(grid))
    for j in 1:Ny, i in 1:Nx
        Wu = Δxᶠᶜᶜ(i, j, 1, grid) * Axᶠᶜᶜ(i, j, 1, grid) * U.u[i, j, 1] * Fu(i, j)
        Wv = Δyᶜᶠᶜ(i, j, 1, grid) * Ayᶜᶠᶜ(i, j, 1, grid) * U.v[i, j, 1] * Fv(i, j)
        work += Wu + Wv
        scale += abs(Wu) + abs(Wv)
    end
    return abs(work) / scale
end

relative_coriolis_work(grid, coriolis, U) = relative_work(grid, U, (i, j) -> x_f_cross_U(i, j, 1, grid, coriolis, U),
                                                                   (i, j) -> y_f_cross_U(i, j, 1, grid, coriolis, U))

relative_vorticity_work(grid, advection, U) = relative_work(grid, U, (i, j) -> horizontal_advection_U(i, j, 1, grid, advection, U.u, U.v),
                                                                     (i, j) -> horizontal_advection_V(i, j, 1, grid, advection, U.u, U.v))

function test_coriolis_work_with_partial_cells(FT)
    rectilinear = RectilinearGrid(CPU(), FT; size=(16, 12, 1), halo=(3, 3, 3), x=(0, 1e6), y=(0, 1e6), z=(-4000, 0),
                                  topology=(Periodic, Periodic, Bounded))

    latlon = LatitudeLongitudeGrid(CPU(), FT; size=(16, 12, 1), halo=(3, 3, 3), longitude=(0, 360), latitude=(-60, 60),
                                   z=(-4000, 0))

    stretched_latitude = [-60 + 120 * (j / 12)^(3/2) for j in 0:12]
    stretched_latlon = LatitudeLongitudeGrid(CPU(), FT; size=(16, 12, 1), halo=(3, 3, 3), longitude=(0, 360),
                                             latitude=stretched_latitude, z=(-4000, 0))

    for (underlying_grid, coriolis) in ((rectilinear, scheme -> FPlane(FT; f=1e-4, scheme)),
                                        (latlon, scheme -> HydrostaticSphericalCoriolis(FT; scheme)),
                                        (stretched_latlon, scheme -> HydrostaticSphericalCoriolis(FT; scheme)))
        grid = partial_cell_grid(underlying_grid)
        U = random_masked_velocities(grid)

        # Exact up to roundoff for EnergyConserving, which uses the thickness-weighted flux on partial cells
        @test relative_coriolis_work(grid, coriolis(EnergyConserving()), U) < 100 * eps(FT)

        # Not for EnstrophyConserving, whose work vanishes only for uniform Δz
        @test relative_coriolis_work(grid, coriolis(EnstrophyConserving()), U) > 1e-3

        # The VectorInvariant vorticity term uses the same flux with q = ζ / h
        @test relative_vorticity_work(grid, VectorInvariant(vorticity_scheme=EnergyConserving()), U) < 100 * eps(FT)
    end
end

#####
##### 11. With uniform Δz, EnergyConserving on a PartialCellBottom grid matches the GridFittedBottom grid
#####

function test_partial_cell_energy_conserving_uniform_Δz(FT)
    underlying_grid = LatitudeLongitudeGrid(CPU(), FT; size=(8, 8, 2), latitude=(30, 60), longitude=(0, 30), z=(-100, 0))
    Nx, Ny, _ = size(underlying_grid)
    bottom_height = fill(FT(-200), Nx, Ny)
    bottom_height[1:3, 1:2] .= 0 # some land

    coriolis = HydrostaticSphericalCoriolis(FT, scheme=EnergyConserving())

    grid_fitted_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))
    partial_cell_grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height))

    U_grid_fitted = random_masked_velocities(grid_fitted_grid)
    U_partial_cell = random_masked_velocities(partial_cell_grid)

    for j in 1:Ny, i in 1:Nx
        @test x_f_cross_U(i, j, 1, partial_cell_grid, coriolis, U_partial_cell) ≈ x_f_cross_U(i, j, 1, grid_fitted_grid, coriolis, U_grid_fitted)
        @test y_f_cross_U(i, j, 1, partial_cell_grid, coriolis, U_partial_cell) ≈ y_f_cross_U(i, j, 1, grid_fitted_grid, coriolis, U_grid_fitted)
    end
end

#####
##### 12. Spurious growing mode with a single-level PartialCellBottom (issue #6397)
#####
##### Unforced, inviscid, f-plane, explicit free surface, random initial velocities over random depths
##### from 8 m to 4000 m. With EnstrophyConserving the energy grows exponentially; with EnergyConserving,
##### which uses the thickness-weighted flux on partial cells, it does not grow. Nor does it with
##### VectorInvariant momentum advection, whose vorticity term uses the same flux.
#####

function total_energy(model)
    grid = model.grid
    g = model.free_surface.gravitational_acceleration
    u, v = interior(model.velocities.u), interior(model.velocities.v)
    η = interior(model.free_surface.displacement)
    Nx, Ny, _ = size(grid)
    return sum(Vᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]^2 / 2 +
               Vᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]^2 / 2 +
               g * Azᶜᶜᶜ(i, j, 1, grid) * η[i, j, 1]^2 / 2 for i in 1:Nx, j in 1:Ny)
end

function energy_growth_with_partial_cells(scheme, timestepper; N=24, steps=500, Δt=600, amplitude=0.05,
                                          gravitational_acceleration=0.01, momentum_advection=nothing)
    underlying_grid = RectilinearGrid(CPU(); size=(N, N, 1), halo=(3, 3, 3), x=(0, 1e6), y=(0, 1e6), z=(-4000, 0),
                                      topology=(Periodic, Periodic, Bounded))

    grid = partial_cell_grid(underlying_grid)

    model = HydrostaticFreeSurfaceModel(grid; timestepper, coriolis = FPlane(; f=1e-4, scheme),
                                        free_surface = ExplicitFreeSurface(; gravitational_acceleration),
                                        momentum_advection, buoyancy = nothing, tracers = nothing, closure = nothing)

    Random.seed!(1)
    set!(model, u = amplitude .* randn(N, N, 1), v = amplitude .* randn(N, N, 1))
    E₀ = total_energy(model)
    run!(Simulation(model; Δt, stop_iteration=steps, verbose=false))

    return total_energy(model) / E₀
end

#####
##### 1. Instantiation tests for new scheme types
#####

for arch in archs
    @testset "Coriolis scheme instantiation" begin
        for FT in float_types
            coriolis = SphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())
            @test coriolis.scheme isa ActiveWeightedEnstrophyConserving

            coriolis = SphericalCoriolis(FT, scheme=ActiveWeightedEnergyConserving())
            @test coriolis.scheme isa ActiveWeightedEnergyConserving

            coriolis = SphericalCoriolis(FT, scheme=TriadScheme())
            @test coriolis.scheme isa TriadScheme

            coriolis = HydrostaticSphericalCoriolis(FT, scheme=ActiveWeightedEnstrophyConserving())
            @test coriolis.scheme isa ActiveWeightedEnstrophyConserving

            coriolis = HydrostaticSphericalCoriolis(FT, scheme=TriadScheme())
            @test coriolis.scheme isa TriadScheme

            # Default scheme for HydrostaticSphericalCoriolis is EnstrophyConserving
            coriolis = HydrostaticSphericalCoriolis(FT)
            @test coriolis.scheme isa EnstrophyConserving
        end
    end

    @testset "Coriolis scheme stencil correctness" begin
        for FT in float_types
            @testset "EnstrophyConserving uniform velocity [$FT]" begin
                test_enstrophy_conserving_uniform_v(FT)
                test_enstrophy_conserving_uniform_u(FT)
            end

            @testset "ActiveWeighted = plain on flat bottom [$FT]" begin
                test_active_weighted_equals_plain_flat_bottom(FT)
            end

            @testset "Immersed boundary Coriolis (Jamart correction)" begin
                test_jamart_correction_near_topography(FT)
            end

            @testset "EEN triad structure [$FT]" begin
                test_een_triad_structure(FT)
            end

            @testset "Coriolis and vorticity work with partial cells [$FT]" begin
                test_coriolis_work_with_partial_cells(FT)
            end

            @testset "EnergyConserving on PartialCellBottom with uniform Δz [$FT]" begin
                test_partial_cell_energy_conserving_uniform_Δz(FT)
            end
        end
    end

    for FT in float_types, scheme in (EnstrophyConserving(),
                                      EnergyConserving(),
                                      ActiveWeightedEnstrophyConserving(),
                                      ActiveWeightedEnergyConserving(),
                                      TriadScheme())

        @testset "Antisymmetry [$FT]" begin
            @testset "scheme=$(summary(scheme))" begin
                test_coriolis_antisymmetry(FT, scheme)
            end
        end

        @testset "Geostrophic balance" begin
            @testset "scheme=$(summary(scheme))" begin
                test_geostrophic_balance_steady(FT, arch, scheme)
            end
        end


        @testset "Energy conservation" begin
            @testset "scheme=$(summary(scheme))" begin
                test_coriolis_energy_conservation(FT, arch, scheme)
            end
        end

        @testset "Inertial oscillation (f-plane)" begin
            @testset "scheme=$(summary(scheme))" begin
                test_inertial_oscillation(FT, arch, scheme)
            end
        end
    end

    if arch isa CPU
        for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
            @testset "Spurious growing mode with single-level PartialCellBottom (#6397) [$timestepper]" begin
                @test energy_growth_with_partial_cells(EnstrophyConserving(), timestepper) > 2
                @test energy_growth_with_partial_cells(EnergyConserving(), timestepper) < 1.01
            end

            @testset "No growing mode with VectorInvariant vorticity and single-level PartialCellBottom [$timestepper]" begin
                momentum_advection = VectorInvariant(vorticity_scheme=EnergyConserving())
                @test energy_growth_with_partial_cells(EnergyConserving(), timestepper; momentum_advection,
                                                       gravitational_acceleration=9.81, amplitude=2, Δt=30, steps=4000) < 1
            end
        end
    end

    for FT in float_types, timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        @testset "DualGridScheme inertial oscillations [$FT, $timestepper]" begin
            test_dual_grid_scheme_inertial_oscillations(FT, arch, timestepper)
        end

        @testset "DualGridScheme checkpoint restart [$FT, $timestepper]" begin
            test_dual_grid_scheme_checkpoint_restart(FT, arch, timestepper)
        end
    end
end
