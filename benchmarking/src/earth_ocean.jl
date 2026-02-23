#####
##### Earth ocean benchmark case
#####
##### Global ocean simulation using HydrostaticFreeSurfaceModel
##### with a TripolarGrid and realistic Earth bathymetry from NumericalEarth.
#####

using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

"""
    earth_ocean(arch = CPU();
                float_type = Float32,
                Nx = 360, Ny = 180, Nz = 50,
                grid_type = "tripolar",
                momentum_advection = WENOVectorInvariant(order=9),
                tracer_advection = WENO(order=7),
                closure = CATKEVerticalDiffusivity(),
                timestepper = :SplitRungeKutta3,
                tracers = (:T, :S))

Create a `HydrostaticFreeSurfaceModel` for the Earth ocean benchmark case
with realistic Earth bathymetry from NumericalEarth.

# Arguments
- `arch`: Architecture to run on (`CPU()` or `GPU()`)

# Keyword Arguments
- `float_type`: Floating point precision (`Float32` or `Float64`)
- `Nx, Ny, Nz`: Grid resolution (longitude, latitude, vertical)
- `grid_type`: `"tripolar"` for a TripolarGrid, `"lat_lon"` for a LatitudeLongitudeGrid with bathymetry, or `"lat_lon_flat"` for a plain LatitudeLongitudeGrid without bathymetry
- `momentum_advection`: Momentum advection scheme (default: `WENOVectorInvariant(order=9)`)
- `tracer_advection`: Tracer advection scheme (default: `WENO(order=7)`)
- `closure`: Turbulence closure (default: `CATKEVerticalDiffusivity()`)
- `timestepper`: Time stepping scheme (default: `:SplitRungeKutta3`)
- `tracers`: Tuple of tracer names (default: `(:T, :S)`)
"""
function earth_ocean(arch = CPU();
                     float_type = Float32,
                     Nx = 360, Ny = 180, Nz = 50,
                     grid_type = "tripolar",
                     momentum_advection = WENOVectorInvariant(order=9),
                     tracer_advection = WENO(order=7),
                     closure = CATKEVerticalDiffusivity(),
                     timestepper = :SplitRungeKutta3,
                     tracers = (:T, :S))

    grid_type in ("tripolar", "lat_lon", "lat_lon_flat") ||
        error("Unknown grid_type: $grid_type. Use \"tripolar\", \"lat_lon\", or \"lat_lon_flat\".")

    Oceananigans.defaults.FloatType = float_type

    depth = 5000  # meters
    z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)

    if grid_type == "tripolar"
        underlying_grid = TripolarGrid(arch;
            size = (Nx, Ny, Nz),
            halo = (7, 7, 7),
            z
        )
    else # lat_lon or lat_lon_flat
        underlying_grid = LatitudeLongitudeGrid(arch;
            size = (Nx, Ny, Nz),
            halo = (7, 7, 7),
            longitude = (0, 360),
            latitude = (-80, 85),
            z
        )
    end

    if grid_type == "lat_lon_flat"
        grid = underlying_grid
    else
        bottom_height = NumericalEarth.regrid_bathymetry(underlying_grid;
            minimum_depth = 10,
            interpolation_passes = 10,
            major_basins = 2
        )

        grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom_height);
            active_cells_map = true
        )
    end

    free_surface = SplitExplicitFreeSurface(; substeps=30)
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    coriolis = HydrostaticSphericalCoriolis()

    model = HydrostaticFreeSurfaceModel(grid;
        momentum_advection,
        tracer_advection,
        coriolis,
        buoyancy,
        closure,
        free_surface,
        tracers,
        timestepper
    )

    # Initial conditions: baroclinic wave excitation
    Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
    Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()

    ic = Dict{Symbol,Any}()
    :T in tracers && (ic[:T] = Tᵢ)
    :S in tracers && (ic[:S] = Sᵢ)
    # Extra tracers beyond T/S get zero (default)

    set!(model; ic...)

    return model
end
