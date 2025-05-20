using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZCoordinate

arch = CPU()
with_wind = true

# function run_tripolar_test_simulation(arch; with_wind = false)

    z_stretched = MutableVerticalDiscretization((-20, 0)) # collect(-20:0) # 
    grid = TripolarGrid(arch; size = (20, 20, 20), z = z_stretched, halo = (7, 7, 7))
    # grid = LatitudeLongitudeGrid(arch; size = (20, 20, 20), latitude = (-80, 80), longitude = (0, 360), z = z_stretched)
    
    function mtn₁(λ, φ)
        λ₁ = 70
        φ₁ = 55
        dφ = 5
        return exp(-((λ - λ₁)^2 + (φ - φ₁)^2) / 2dφ^2)
    end

    function mtn₂(λ, φ)
        λ₁ = 70
        λ₂ = λ₁ + 180
        φ₂ = 55
        dφ = 5
        return exp(-((λ - λ₂)^2 + (φ - φ₂)^2) / 2dφ^2)
    end

    zb = - 20
    h  = - zb + 10
    gaussian_islands(λ, φ) = zb + h * (mtn₁(λ, φ) + mtn₂(λ, φ))
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))
    free_surface = SplitExplicitFreeSurface(grid; substeps=10)
    # free_surface = ExplicitFreeSurface()

    kwargs = (;
        grid,
        free_surface,
        tracer_advection = UpwindBiased(order=1),
        tracers = (:b, :c),
        # timestepper = :SplitRungeKutta3,
        buoyancy = BuoyancyTracer(),
        vertical_coordinate = ifelse(grid.z isa MutableVerticalDiscretization, ZStar(), ZCoordinate()),
    )

    if with_wind
        model = HydrostaticFreeSurfaceModel(; kwargs...) #, boundary_conditions = (u = u_bcs, v = v_bcs))   
    else
        model = HydrostaticFreeSurfaceModel(; kwargs...)
    end
    
    bᵢ(x, y, z) = x < 120 && y > 60 ? 1.0 : 0.0

    set!(model, c = (x, y, z) -> 1.0, b = bᵢ) 

    simulation = Simulation(model, Δt=2minutes, stop_time=1day)
    
    c = simulation.model.tracers.c
    tracer_integral = Integral(c, dims=(1, 2, 3))

    if with_wind
        prefix = "test_with_wind"
    else
        prefix = "test_no_wind"
    end
    
    integral_ow = JLD2Writer(
        simulation.model,
        (; tracer_integral),
        filename = "$(prefix)_integrals",
        schedule = IterationInterval(1),
        overwrite_existing = true,
    )

    simulation.output_writers[:integral] = integral_ow

    run!(simulation)
    
    Oceananigans.ImmersedBoundaries.mask_immersed_field!(model.tracers.c, 1.0)
    # return nothing
# end

# run_tripolar_test_simulation(CPU(); with_wind = false)
# run_tripolar_test_simulation(CPU(); with_wind = true)

iw = FieldTimeSeries("test_with_wind_integrals.jld2", "tracer_integral")
# in = FieldTimeSeries("test_no_wind_integrals.jld2", "tracer_integral")