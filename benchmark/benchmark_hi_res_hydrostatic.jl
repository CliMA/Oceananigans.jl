using Oceananigans
using SeawaterPolynomials: TEOS10EquationOfState
using CUDA

function many_steps!(model, Nt; Δt=1e-3)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
end

function hi_res_hydrostatic_model(grid;
    vertical_coordinate = ZCoordinate(),
    momentum_advection = WENOVectorInvariant(order=9),
    passive_tracers = (),
    tracer_advection = WENO(order=7),
    closure = CATKEVerticalDiffusivity(),
    timestepper = :SplitRungeKutta3)
    
    tracers = tuple(:T, :S, :e, passive_tracers...)
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    free_surface = SplitExplicitFreeSurface(substeps=30)
    coriolis = HydrostaticSphericalCoriolis()

    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection, coriolis,
                                        buoyancy, closure, free_surface, tracers, vertical_coordinate, timestepper)

    # Sensible initial condition
    Ξ(x, y, z) = rand()
    dTdz = 1 / grid.Lz
    Tᵢ(x, y, z) = 20 + dTdz * z + 1e-4 * Ξ(x, y, z)
    Sᵢ(x, y, z) = 35 + 1e-4 * Ξ(x, y, z)
    uᵢ(x, y, z) = 1e-3 * Ξ(x, y, z)
    
    set!(model, T=Tᵢ, S=Sᵢ, u=uᵢ, v=uᵢ)

    return model
end

# Configurations
# group = get(ENV, "BENCHMARK_GROUP", "all") |> Symbol

config = :channel
Nx = 512
Ny = 256
Nz = 128
arch = GPU()
immersed = false
Oceananigans.defaults.FloatType = Float64

lat_lon_kw = (longitude=(0, 360), latitude=(-80, 80), z=(-1000, 0), size=(Nx, Ny, Nz), halo=(7, 7, 7))
dλ = 20 # ridge width in degrees
ridge(λ, φ) = -1000 + 800 * exp(-(λ - 30)^2 / 2dλ^2)

if config == :channel
    grid = LatitudeLongitudeGrid(arch; topology=(Periodic, Bounded, Bounded), lat_lon_kw...)

    if immersed
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(ridge))
    end

    model = hi_res_hydrostatic_model(grid)
elseif config == :box

    Oceananigans.defaults.FloatType = FT
    grid = LatitudeLongitudeGrid(arch; topology=(Bounded, Bounded, Bounded), lat_lon_kw...)

    if immersed
        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(ridge))
    end

    model = hi_res_hydrostatic_model(grid)
end

@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, 10)
