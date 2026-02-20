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
                momentum_advection = WENOVectorInvariant(order=9),
                tracer_advection = WENO(order=7),
                closure = CATKEVerticalDiffusivity(),
                timestepper = :QuasiAdamsBashforth2)

Create a `HydrostaticFreeSurfaceModel` for the Earth ocean benchmark case.

Uses a TripolarGrid with realistic Earth bathymetry from NumericalEarth.

# Arguments
- `arch`: Architecture to run on (`CPU()` or `GPU()`)

# Keyword Arguments
- `float_type`: Floating point precision (`Float32` or `Float64`)
- `Nx, Ny, Nz`: Grid resolution (longitude, latitude, vertical)
- `momentum_advection`: Momentum advection scheme (default: `WENOVectorInvariant(order=9)`)
- `tracer_advection`: Tracer advection scheme (default: `WENO(order=7)`)
- `closure`: Turbulence closure (default: `CATKEVerticalDiffusivity()`)
- `timestepper`: Time stepping scheme (default: `:QuasiAdamsBashforth2`)
"""
function earth_ocean(arch = CPU();
                     float_type = Float32,
                     Nx = 360, Ny = 180, Nz = 50,
                     momentum_advection = WENOVectorInvariant(order=9),
                     tracer_advection = WENO(order=7),
                     closure = CATKEVerticalDiffusivity(),
                     timestepper = :QuasiAdamsBashforth2)

    Oceananigans.defaults.FloatType = float_type

    depth = 5000  # meters
    z = ExponentialDiscretization(Nz, -depth, 0; scale=depth/4)

    underlying_grid = TripolarGrid(arch;
        size = (Nx, Ny, Nz),
        halo = (7, 7, 7),
        z
    )

    bottom_height = NumericalEarth.regrid_bathymetry(underlying_grid;
        minimum_depth = 10,
        interpolation_passes = 10,
        major_basins = 2
    )

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
        active_cells_map = true
    )

    tracers = (:T, :S)

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
    set!(model, T=Tᵢ, S=Sᵢ)

    return model
end
