using Oceananigans
using SeawaterPolynomials: TEOS10EquationOfState
using NVTX
using CUDA
using ClimaOcean

function many_steps!(model, Nt; Δt=1e-3)
    for _ in 1:Nt
        NVTX.@range "time step" begin
            time_step!(model, Δt)
        end
    end
end

function hi_res_hydrostatic_model(grid;
    vertical_coordinate = ZCoordinate(),
    passive_tracers = (),
    momentum_advection = WENOVectorInvariant(order=9),
    tracer_advection = WENO(order=7),
    closure = CATKEVerticalDiffusivity(),
    timestepper = :QuasiAdamsBashforth2)
    
    tracers = tuple(:T, :S, :e, passive_tracers...)
    buoyancy = SeawaterBuoyancy(equation_of_state=TEOS10EquationOfState())
    free_surface = SplitExplicitFreeSurface(substeps=30)
    coriolis = HydrostaticSphericalCoriolis()

    model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, tracer_advection, coriolis,
                                        buoyancy, closure, free_surface, tracers,
                                        vertical_coordinate, timestepper)

    # Initial condition that excites a baroclinic wave
    Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
    Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()
    set!(model, T=Tᵢ, S=Sᵢ)
    
    return model
end

function latitude_longitude_grid(arch, Nx, Ny, Nz; immersed=true, halo=(7, 7, 7), kw...)
    grid = LatitudeLongitudeGrid(arch; topology=(Periodic, Bounded, Bounded), halo, size=(Nx, Ny, Nz), kw...)

    if immersed
        dλ = 20 # ridge width in degrees
        ridge(λ, φ) = -1000 + 800 * exp(-(λ - 30)^2 / 2dλ^2)
        grid = ImmersedBoundaryGrid(grid, PartialCellBottom(ridge))
    end

    return grid
end

"""
    tripolar_grid(arch, Nx, Ny, Nz; halo=(7, 7, 7), kw...)

Return a tripolar grid with Gaussian islands over the two north poles.
"""
function tripolar_grid(arch, Nx, Ny, Nz; bathy=false, halo=(7, 7, 7), kw...)
    underlying_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo, z)
    bathymetry = Nothing

    if bathy
        bathymetry = ClimaOcean.regrid_bathymetry(underlying_grid)
    else
        H = - z[2]
        dφ, dλ = 4, 8
        λ₀, φ₀ = 70, 55
        h = 100
        
        isle(λ, φ) = exp(-(λ - λ₀)^2 / 2dλ^2 - (φ - φ₀)^2 / 2dφ^2)
        gaussian_isles(λ, φ) = - H + (H + h) * (isle(λ, φ) + isle(λ - 180, φ))
	    bathymetry = gaussian_isles
    end
    
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

    return grid
end

# Configurations
# group = get(ENV, "BENCHMARK_GROUP", "all") |> Symbol

config = :tripolar
immersed = true # note :tripolar is always immersed
Nx = 512
Ny = 256
Nz = 128
arch = GPU()
Oceananigans.defaults.FloatType = Float64
Nt = 100
z = (-3000, 0)
lat_lon_kw = (; longitude=(0, 360), latitude=(-80, 80), z)
tripolar_kw = (; z)

model_kw = (;
    # momentum_advection = nothing,
    # tracer_advection = nothing,
    # momentum_advection = nothing,
    tracer_advection = WENO(order=7),
)

if config == :tripolar
    grid = tripolar_grid(arch, Nx, Ny, Nz; bathy=true, tripolar_kw...)
    model = hi_res_hydrostatic_model(grid; model_kw...)

elseif config == :channel
    grid = latitude_longitude_grid(arch, Nx, Ny, Nz; immersed, topology=(Periodic, Bounded, Bounded), lat_lon_kw...)
    model = hi_res_hydrostatic_model(grid; model_kw...)

elseif config == :box
    grid = latitude_longitude_grid(arch, Nx, Ny, Nz; immersed, topology=(Bounded, Bounded, Bounded), lat_lon_kw...)
    model = hi_res_hydrostatic_model(grid; model_kw...)
end

@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, 1) # compile
@time many_steps!(model, Nt)
