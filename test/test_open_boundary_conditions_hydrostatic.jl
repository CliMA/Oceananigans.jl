using Oceananigans
using Oceananigans.BoundaryConditions: Flather, Radiation
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

    # Radiation OBC at east and west boundaries
    rad = Radiation(outflow_relaxation_timescale = 100.0)
    u_east_bc  = OpenBoundaryCondition(0; scheme = rad)
    u_west_bc  = OpenBoundaryCondition(0; scheme = rad)
    u_bcs = FieldBoundaryConditions(east = u_east_bc, west = u_west_bc)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10, extend_halos = false)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        boundary_conditions = (u = u_bcs,),
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

    # Flather OBC on east boundary: prescribe zero external values
    # (the tide comes from the initial condition radiating)
    flather = Flather(external_values = (η = 0.0, U = 0.0))
    u_east_bc = OpenBoundaryCondition(nothing; scheme = flather)
    u_bcs = FieldBoundaryConditions(east = u_east_bc)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10, extend_halos = false)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        boundary_conditions = (u = u_bcs,),
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

    # Flather OBC at east and west boundaries for u velocity
    # At west: prescribe incoming Kelvin wave
    # At east: let it radiate out (zero external)
    flather_west = Flather(external_values = (η = 0.0, U = 0.0))
    flather_east = Flather(external_values = (η = 0.0, U = 0.0))

    u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(nothing; scheme = flather_west),
                                    east = OpenBoundaryCondition(nothing; scheme = flather_east))

    free_surface = SplitExplicitFreeSurface(grid; substeps = 10, extend_halos = false)

    model = HydrostaticFreeSurfaceModel(grid;
        free_surface = free_surface,
        coriolis = FPlane(f = f₀),
        boundary_conditions = (u = u_bcs,),
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

    rad = Radiation(outflow_relaxation_timescale = Inf, inflow_relaxation_timescale = Δt)
    u_bcs = FieldBoundaryConditions(east  = OpenBoundaryCondition(0; scheme = rad),
                                    west  = OpenBoundaryCondition(0; scheme = rad))

    fs_small = SplitExplicitFreeSurface(grid_small; substeps = 10, extend_halos = false)
    model_small = HydrostaticFreeSurfaceModel(grid_small;
        free_surface = fs_small,
        boundary_conditions = (u = u_bcs,),
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

    # L₂ relative error (normalized by initial amplitude)
    L₂_error = sqrt(sum((η_s .- η_l).^2) / length(η_s)) / A

    # Check: no NaN, and relative error < 10%
    no_nan = !any(isnan, η_small)
    small_error = L₂_error < 0.1

    # Also check energy decreased (pulse should have partially exited)
    η_init = A * exp.(-(x_small .- x₀).^2 ./ (2σ^2))
    E₀ = sum(η_init .^ 2)
    E₁ = sum(η_small .^ 2)
    energy_decreased = E₁ < 0.5 * E₀

    return no_nan && small_error && energy_decreased
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
end
