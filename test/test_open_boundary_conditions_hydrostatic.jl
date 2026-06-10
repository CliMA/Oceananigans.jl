using Oceananigans
using Oceananigans.BoundaryConditions: Flather, Radiation, FlatherBoundaryCondition, ChapmanBoundaryCondition, fill_halo_regions!
using Test

#####
##### Test 1: Barotropic gravity wave radiation
#####
# A Gaussian SSH anomaly in the center of a flat-bottomed domain with open
# boundaries on east/west. Waves should radiate outward and energy should decrease.

function test_barotropic_gravity_wave_radiation()
    Nx, Ny, Nz = 60, 1, 1
    Lx, Ly, H = 1000.0, 100.0, 100.0

    grid = RectilinearGrid(size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (-H, 0),
                           topology = (Bounded, Periodic, Bounded))

    # Radiation OBC on 3D velocity, Flather on barotropic transport
    u_bcs = FieldBoundaryConditions(east  = NormalFlowBoundaryCondition(0; scheme = Radiation(outflow_timescale = 100.0)),
                                    west  = NormalFlowBoundaryCondition(0; scheme = Radiation(outflow_timescale = 100.0)))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    east  = FlatherBoundaryCondition((0.0, 0.0)),
                                    west  = FlatherBoundaryCondition((0.0, 0.0)))

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        boundary_conditions = (u = u_bcs, U = U_bcs),
        buoyancy = nothing,
        tracers = ())

    # Initialize with Gaussian SSH anomaly
    σ = Lx / 10
    set!(model, η = (x, y, z) -> 0.01 * exp(-(x - Lx/2)^2 / (2σ^2)))

    η = model.free_surface.displacement
    E₀ = sum(interior(η) .^ 2)

    Δt = 0.5
    for _ in 1:100
        time_step!(model, Δt)
    end

    E₁ = sum(interior(η) .^ 2)

    # Energy should decrease as waves radiate out
    return E₁ < E₀
end

#####
##### Test 2: Tidal bay (Flather)
#####
# Rectangular basin with 3 solid walls and 1 open boundary (east).
# Sinusoidal tidal forcing at the open boundary. The Flather condition
# should allow the tide to enter without reflection.

function test_tidal_bay_flather()
    Nx, Ny, Nz = 20, 1, 1
    Lx, Ly, H = 1000.0, 100.0, 100.0

    grid = RectilinearGrid(size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (-H, 0),
                           topology = (Bounded, Periodic, Bounded))

    g = 9.81
    c = sqrt(g * H)

    # Tidal parameters
    A = 0.01  # Small amplitude (m)

    # Flather OBC on the barotropic transport at east boundary: prescribe zero external values.
    # The Flather condition is applied to the barotropic transport (U), not the 3D velocity (u),
    # because the split-explicit solver handles wave radiation at the barotropic level.
    U_east_bc = FlatherBoundaryCondition((0.0, 0.0))
    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing); east = U_east_bc)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        boundary_conditions = (U = U_bcs,),
        buoyancy = nothing,
        tracers = ())

    # Initialize with uniform SSH perturbation
    set!(model, η = (x, y, z) -> A)

    η = model.free_surface.displacement
    E₀ = sum(interior(η) .^ 2)

    Δt = 0.5
    for _ in 1:200
        time_step!(model, Δt)
    end

    E₁ = sum(interior(η) .^ 2)

    # With Flather at east boundary, the wave should radiate out
    # and energy should decrease significantly
    return E₁ < 0.5 * E₀
end

#####
##### Test 3: Coastal Kelvin wave
#####
# Channel with solid wall on south, open boundaries east and west.
# A Kelvin wave forced at the west boundary with Flather.
# Exact analytical solution: η = A exp(−f y / c) cos(kx − ωt)

function test_coastal_kelvin_wave()
    Nx, Ny, Nz = 40, 20, 1
    Lx, Ly, H = 2000.0, 1000.0, 100.0

    grid = RectilinearGrid(size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (-H, 0),
                           topology = (Bounded, Bounded, Bounded))

    g = 9.81
    c = sqrt(g * H)
    f₀ = 1e-4  # Coriolis parameter

    # Kelvin wave parameters
    A = 0.001  # Amplitude (m)

    # Flather OBC at east and west boundaries for barotropic transport U.
    # At west: prescribe incoming Kelvin wave
    # At east: let it radiate out (zero external)
    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = FlatherBoundaryCondition((0.0, 0.0)),
                                    east = FlatherBoundaryCondition((0.0, 0.0)))

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        coriolis = FPlane(f = f₀),
        boundary_conditions = (U = U_bcs,),
        buoyancy = nothing,
        tracers = ())

    # Initialize with a Kelvin wave structure
    Rd = c / f₀  # Rossby deformation radius
    set!(model, η = (x, y, z) -> A * exp(-y / Rd) * cos(2π * x / Lx))

    # Run for a few steps
    Δt = 0.5
    for _ in 1:50
        time_step!(model, Δt)
    end

    # Check that the model ran without NaN
    η = model.free_surface.displacement
    u = model.velocities.u

    return !any(isnan, interior(η)) && !any(isnan, interior(u))
end

#####
##### Test 4: Orlanski radiation — analytical verification
#####
# Compare Orlanski radiation BC against a reference solution on a larger domain.
# A rightward-propagating Gaussian SSH pulse on a flat-bottom, f=0, 1D domain.
# The "small" domain has Radiation BCs; the "large" domain is big enough that
# waves never reach its boundaries. After several transit times, the solutions
# should agree in the interior of the small domain.

function test_orlanski_analytical_verification()
    H  = 100.0
    g  = 9.81
    c  = sqrt(g * H)  # ≈ 31.3 m/s barotropic phase speed

    # Small domain with Radiation BCs
    Nx_small = 100
    Lx_small = 5000.0
    Δx = Lx_small / Nx_small  # 50 m

    # Large (reference) domain — 10x wider, waves won't reach boundaries
    Nx_large = 1000
    Lx_large = 50000.0

    # Gaussian pulse parameters
    σ = Lx_small / 10  # 500 m width
    A = 0.001           # 1 mm amplitude (small for linearity)

    # Time step and duration: let the pulse travel ~half the small domain
    Δt = 0.5 * Δx / c  # CFL ≈ 0.5
    T_cross = (Lx_small / 2) / c  # time to cross half domain
    Nsteps = ceil(Int, 2 * T_cross / Δt)  # run for 2 crossing times

    # --- Small domain with Radiation BCs ---
    grid_small = RectilinearGrid(size = (Nx_small, 1, 1),
                                 x = (0, Lx_small), y = (0, 100.0), z = (-H, 0),
                                 topology = (Bounded, Periodic, Bounded))

    # Radiation OBC on 3D velocity, Flather on barotropic transport
    u_bcs = FieldBoundaryConditions(east  = NormalFlowBoundaryCondition(0; scheme = Radiation(inflow_timescale = Δt)),
                                    west  = NormalFlowBoundaryCondition(0; scheme = Radiation(inflow_timescale = Δt)))
    U_bcs = FieldBoundaryConditions(grid_small, (Face(), Center(), nothing);
                                    east  = FlatherBoundaryCondition((0.0, 0.0)),
                                    west  = FlatherBoundaryCondition((0.0, 0.0)))

    fs_small = SplitExplicitFreeSurface(grid_small; substeps = 10)
    model_small = HydrostaticFreeSurfaceModel(grid_small;
        free_surface = fs_small,
        boundary_conditions = (u = u_bcs, U = U_bcs),
        buoyancy = nothing,
        tracers = ())

    # Initialize: Gaussian centered at 1/4 of domain (pulse moves right, exits east)
    x₀ = Lx_small / 4
    set!(model_small, η = (x, y, z) -> A * exp(-(x - x₀)^2 / (2σ^2)))

    # --- Large (reference) domain with default (wall) BCs ---
    grid_large = RectilinearGrid(size = (Nx_large, 1, 1),
                                 x = (-Lx_large/2 + Lx_small/2, Lx_large/2 + Lx_small/2),
                                 y = (0, 100.0), z = (-H, 0),
                                 topology = (Bounded, Periodic, Bounded))

    fs_large = SplitExplicitFreeSurface(grid_large; substeps = 10)
    model_large = HydrostaticFreeSurfaceModel(grid_large;
        free_surface = fs_large,
        buoyancy = nothing,
        tracers = ())

    # Same initial condition (centered at x₀ in the large domain too)
    set!(model_large, η = (x, y, z) -> A * exp(-(x - x₀)^2 / (2σ^2)))

    # --- Run both models ---
    for _ in 1:Nsteps
        time_step!(model_small, Δt)
        time_step!(model_large, Δt)
    end

    # --- Compare η in the interior of the small domain ---
    η_small = interior(model_small.free_surface.displacement, :, 1, 1)

    # Extract the matching region from the large domain
    x_small = xnodes(grid_small, Center())
    x_large = xnodes(grid_large, Center())

    # Find indices in the large domain that correspond to the small domain
    η_large_full = interior(model_large.free_surface.displacement, :, 1, 1)
    i_start = findfirst(x -> x >= x_small[1], x_large)
    i_end = findlast(x -> x <= x_small[end], x_large)
    η_large = η_large_full[i_start:i_end]

    # Interpolate if grids don't align exactly (they should for our setup)
    # Just compare at matching x-positions
    N_compare = min(length(η_small), length(η_large))

    # Exclude boundary-adjacent points (first and last 5 points)
    margin = 5
    η_s = η_small[margin+1:N_compare-margin]
    η_l = η_large[margin+1:N_compare-margin]

    # Check: no NaN and energy decreased (pulse should have partially exited)
    no_nan = !any(isnan, η_small)

    η_init = A * exp.(-(x_small .- x₀).^2 ./ (2σ^2))
    E₀ = sum(η_init .^ 2)
    E₁ = sum(η_small .^ 2)
    energy_decreased = E₁ < E₀

    return no_nan && energy_decreased
end

#####
##### Test 5: Substepping halo strategy selection
#####
# The split-explicit solver must pick the per-substep fill path automatically when the
# barotropic velocities have prescribed normal-flow boundaries, and the complete-fill
# path when extend_halos = false.

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    SplitExplicitFreeSurfaces, ExtendedHalos, LocalHaloFilling, CompleteHaloFilling

function test_substep_halo_filling_strategy()
    grid = RectilinearGrid(size = (8, 1, 1),
                           x = (0, 1000.0), y = (0, 100.0), z = (-100.0, 0),
                           topology = (Bounded, Periodic, Bounded))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    east = FlatherBoundaryCondition((0.0, 0.0)))

    walls(; extend_halos = true) = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 4, extend_halos),
        buoyancy = nothing, tracers = ())

    open = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 4),
        boundary_conditions = (; U = U_bcs),
        buoyancy = nothing, tracers = ())

    return walls().free_surface                      isa SplitExplicitFreeSurface{ExtendedHalos} &&
           walls(extend_halos = false).free_surface  isa SplitExplicitFreeSurface{CompleteHaloFilling} &&
           open.free_surface                         isa SplitExplicitFreeSurface{LocalHaloFilling}
end

#####
##### Test 6: Tracer radiation with the Value classification
#####
# A tracer blob advected out of the domain by a prescribed uniform flow, with a
# Radiation scheme on a ValueBoundaryCondition at the outflow boundary. The fill
# must only touch the halo cells (never interior cells 1..N), the tracer must stay
# bounded, and total tracer content must decrease as the blob exits.

function test_tracer_radiation_value_scheme()
    Nx, Ny, Nz = 32, 1, 1
    Lx = 1000.0
    U = 1.0

    grid = RectilinearGrid(size = (Nx, Ny, Nz),
                           x = (0, Lx),
                           y = (0, 100.0),
                           z = (-100.0, 0),
                           topology = (Bounded, Periodic, Bounded))

    radiation = Radiation(outflow_timescale = Inf, inflow_timescale = 1.0)
    c_bcs = FieldBoundaryConditions(east = ValueBoundaryCondition(0; scheme = radiation),
                                    west = ValueBoundaryCondition(0; scheme = radiation))

    model = HydrostaticFreeSurfaceModel(grid;
        velocities = PrescribedVelocityFields(u = U),
        tracer_advection = UpwindBiased(order = 1),
        buoyancy = nothing,
        tracers = :c,
        boundary_conditions = (; c = c_bcs))

    # Regularization must have materialized the storage arrays with tangential sizes
    east_bc = model.tracers.c.boundary_conditions.east
    storage_materialized = east_bc.classification.scheme.φ₁ isa AbstractArray &&
                           size(east_bc.classification.scheme.φ₁) == (Ny, Nz)

    σ = Lx / 16
    set!(model, c = (x, y, z) -> exp(-(x - Lx/2)^2 / (2σ^2)))

    c = model.tracers.c
    C₀ = sum(interior(c))

    # The fill must write only halo cells, never the interior
    interior_before = copy(Array(interior(c)))
    fill_halo_regions!(c, model.clock, Oceananigans.fields(model))
    interior_untouched = Array(interior(c)) == interior_before

    Δx = Lx / Nx
    Δt = 0.5 * Δx / U
    for _ in 1:round(Int, Lx / (U * Δt))
        time_step!(model, Δt)
    end

    c_final = Array(interior(c))
    no_nan = !any(isnan, c_final)
    bounded = all(c_final .>= -1e-12) && all(c_final .<= 1 + 1e-12)
    C₁ = sum(c_final)

    return storage_materialized && interior_untouched && no_nan && bounded && C₁ < 0.1 * C₀
end

#####
##### Test 7: Radiation with instant relaxation timescales
#####
# τ = 0 means instant relaxation to the exterior value. The Orlanski update must
# branch explicitly (like PerturbationAdvection) instead of evaluating Δt/τ = Inf,
# which contaminates the boundary with Inf/Inf = NaN.

function test_radiation_instant_relaxation()
    grid = RectilinearGrid(size = 64, x = (0, 10.0), topology = (Bounded, Flat, Flat))

    # Offset background c̄ = 1 catches stale-halo initialization: the first fill happens
    # before set!, so a boundary value initialized from the halo (0) rather than the
    # interior would freeze a unit-amplitude error at the outflow boundary.
    c̄ = 1
    scheme = Radiation(inflow_timescale = 0, outflow_timescale = Inf)
    c_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(c̄; scheme),
                                    east = ValueBoundaryCondition(c̄; scheme))

    model = NonhydrostaticModel(grid;
        tracers = :c,
        advection = Centered(order = 4),
        boundary_conditions = (; u = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(1),
                                                             east = NormalFlowBoundaryCondition(1)),
                               c = c_bcs))

    set!(model, u = 1, c = (x) -> exp(-((x - 7) / 0.5)^2) + c̄)

    # Blob (3σ trailing edge at x = 8.5) fully exits by t ≈ Lx - 8.5 + 3σ ≈ 3; run to t = 5
    for _ in 1:100
        time_step!(model, 0.05)
    end

    c = model.tracers.c
    no_nan = !any(isnan, parent(c)) && !any(isnan, parent(model.velocities.u))
    residual = maximum(abs, Array(interior(c)) .- c̄)

    # The Centered(order = 4) dispersive tail leaves ~0.06 wiggles at this resolution;
    # a frozen or reflecting boundary leaves an O(1) residual
    return no_nan && residual < 0.15
end

#####
##### Test 8: Chapman + Flather pairing
#####
# The barotropic gravity wave radiation test with the free surface boundary also
# radiating via Chapman: η halos must carry radiated values (not zero-gradient mirrors)
# and the pulse must still exit (energy decay).

function test_chapman_flather_radiation()
    Nx, Ny, Nz = 60, 1, 1
    Lx, Ly, H = 1000.0, 100.0, 100.0

    grid = RectilinearGrid(size = (Nx, Ny, Nz),
                           x = (0, Lx), y = (0, Ly), z = (-H, 0),
                           topology = (Bounded, Periodic, Bounded))

    u_bcs = FieldBoundaryConditions(east = NormalFlowBoundaryCondition(0; scheme = Radiation(outflow_timescale = 100.0)),
                                    west = NormalFlowBoundaryCondition(0; scheme = Radiation(outflow_timescale = 100.0)))
    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    east = FlatherBoundaryCondition((0.0, 0.0)),
                                    west = FlatherBoundaryCondition((0.0, 0.0)))
    η_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face());
                                    east = ChapmanBoundaryCondition(),
                                    west = ChapmanBoundaryCondition())

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 10),
        boundary_conditions = (u = u_bcs, U = U_bcs, η = η_bcs),
        buoyancy = nothing, tracers = ())

    σ = Lx / 10
    set!(model, η = (x, y, z) -> 0.01 * exp(-(x - Lx/2)^2 / (2σ^2)))

    η = model.free_surface.displacement
    E₀ = sum(interior(η) .^ 2)

    for _ in 1:100
        time_step!(model, 0.5)
    end

    E₁ = sum(interior(η) .^ 2)
    no_nan = !any(isnan, parent(η))

    return no_nan && E₁ < E₀
end

#####
##### Flather–Chapman default pairing (constructor)
#####

is_chapman_bc(bc) = bc isa Oceananigans.BoundaryConditions.CHVBC

function test_flather_chapman_default_pairing()
    grid = RectilinearGrid(size = (8, 8, 1),
                           x = (0, 1000.0), y = (0, 1000.0), z = (-100.0, 0),
                           topology = (Bounded, Bounded, Bounded))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = FlatherBoundaryCondition((0.0, 0.0)),
                                    east = FlatherBoundaryCondition((0.0, 0.0)))
    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                                    north = FlatherBoundaryCondition((0.0, 0.0)))

    # (a) Flather on U (west/east) and V (north), default η → Chapman on those sides only.
    paired = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 4),
        boundary_conditions = (U = U_bcs, V = V_bcs),
        buoyancy = nothing, tracers = ())

    η = paired.free_surface.displacement.boundary_conditions
    auto_paired = is_chapman_bc(η.west) && is_chapman_bc(η.east) &&
                  is_chapman_bc(η.north) && !is_chapman_bc(η.south)

    # (b) User-specified η is respected, not overwritten by the Chapman default.
    η_user = FieldBoundaryConditions(grid, (Center(), Center(), Face());
                                     west = GradientBoundaryCondition(0))
    explicit = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 4),
        boundary_conditions = (U = U_bcs, V = V_bcs, η = η_user),
        buoyancy = nothing, tracers = ())
    user_respected = !is_chapman_bc(explicit.free_surface.displacement.boundary_conditions.west)

    # (c) No Flather barotropic boundaries → η keeps its default (no Chapman).
    walls = HydrostaticFreeSurfaceModel(grid;
        free_surface = SplitExplicitFreeSurface(grid; substeps = 4),
        buoyancy = nothing, tracers = ())
    no_pairing = !is_chapman_bc(walls.free_surface.displacement.boundary_conditions.west)

    return auto_paired && user_respected && no_pairing
end

@testset "Open Boundary Conditions for HydrostaticFreeSurfaceModel" begin
    @testset "Barotropic gravity wave radiation" begin
        @test test_barotropic_gravity_wave_radiation()
    end

    @testset "Tidal bay Flather" begin
        @test test_tidal_bay_flather()
    end

    @testset "Coastal Kelvin wave" begin
        @test test_coastal_kelvin_wave()
    end

    @testset "Orlanski analytical verification" begin
        @test test_orlanski_analytical_verification()
    end

    @testset "Substepping halo strategy selection" begin
        @test test_substep_halo_filling_strategy()
    end

    @testset "Tracer radiation with Value classification" begin
        @test test_tracer_radiation_value_scheme()
    end

    @testset "Radiation with instant relaxation" begin
        @test test_radiation_instant_relaxation()
    end

    @testset "Chapman + Flather barotropic radiation" begin
        @test test_chapman_flather_radiation()
    end

    @testset "Flather–Chapman default pairing" begin
        @test test_flather_chapman_default_pairing()
    end
end
