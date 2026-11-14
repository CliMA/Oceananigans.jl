#####
##### Nonhydrostatic benchmark case
#####
##### Stratified, rotating flow in a box using NonhydrostaticModel,
##### to compare the costs of its pressure solvers.
#####

using Oceananigans.Solvers: ConjugateGradientPoissonSolver

"""
    nonhydrostatic_box(arch = CPU();
                       float_type = Float64,
                       Nx = 64, Ny = 64, Nz = 64,
                       pressure_solver = "FFT")

Create a `NonhydrostaticModel` of a stratified, rotating current in a 1 km × 1 km × 500 m box
that is periodic in x and y. The grid, and with it the pressure solver, depends on `pressure_solver`:

- `"FFT"`: a uniform `RectilinearGrid`, which uses the `FFTBasedPoissonSolver`;
- `"FourierTridiagonal"`: a `RectilinearGrid` stretched in z, which uses the `FourierTridiagonalPoissonSolver`;
- `"ConjugateGradient"`: a uniform `RectilinearGrid` with a seamount (`GridFittedBottom`), which uses the
  `ConjugateGradientPoissonSolver`, preconditioned with the FFT-based solver of the underlying grid.

# Arguments
- `arch`: Architecture to run on (`CPU()` or `GPU()`)

# Keyword Arguments
- `float_type`: Floating point precision (`Float32` or `Float64`)
- `Nx, Ny, Nz`: Grid resolution
- `pressure_solver`: `"FFT"`, `"FourierTridiagonal"`, or `"ConjugateGradient"`
"""
function nonhydrostatic_box(arch = CPU();
                            float_type = Float64,
                            Nx = 64, Ny = 64, Nz = 64,
                            pressure_solver = "FFT")

    pressure_solver in ("FFT", "FourierTridiagonal", "ConjugateGradient") ||
        error("Unknown pressure_solver: $pressure_solver. Use \"FFT\", \"FourierTridiagonal\", or \"ConjugateGradient\".")

    Oceananigans.defaults.FloatType = float_type

    Lx = Ly = 1000 # meters
    Lz = 500       # meters

    z = pressure_solver == "FourierTridiagonal" ? ExponentialDiscretization(Nz, -Lz, 0; scale=Lz/2) : (-Lz, 0)

    # WENO(order=5) on an ImmersedBoundaryGrid needs 4 halo points; all three cases use the same halo
    underlying_grid = RectilinearGrid(arch;
        size = (Nx, Ny, Nz),
        halo = (4, 4, 4),
        x = (0, Lx),
        y = (0, Ly),
        z,
        topology = (Periodic, Periodic, Bounded)
    )

    if pressure_solver == "ConjugateGradient"
        seamount(x, y) = -Lz + 100 * exp(-((x - Lx/2)^2 + (y - Ly/2)^2) / 200^2)
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(seamount))
        solver = ConjugateGradientPoissonSolver(grid)
    else
        grid = underlying_grid
        solver = nothing # the model picks the FFT-based or Fourier-tridiagonal solver for the grid
    end

    model = NonhydrostaticModel(grid;
        pressure_solver = solver,
        advection = WENO(float_type; order=5),
        coriolis = FPlane(float_type; f=1e-4),
        buoyancy = BuoyancyTracer(),
        tracers = :b
    )

    # Initial conditions: a perturbed current in a linear stratification
    U = 0.05  # m/s
    ϵ = 0.005 # m/s
    N² = 1e-5 # s⁻²
    uᵢ(x, y, z) = U + ϵ * sin(6π * x / Lx) * cos(4π * y / Ly) * cos(π * z / Lz)
    vᵢ(x, y, z) = ϵ * cos(4π * x / Lx) * sin(6π * y / Ly) * cos(π * z / Lz)
    bᵢ(x, y, z) = N² * z

    set!(model, u=uᵢ, v=vᵢ, b=bᵢ)

    return model
end
