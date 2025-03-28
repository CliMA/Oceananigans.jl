module PrecompileReactantExtFloat64

using Reactant
using Oceananigans
using PrecompileTools: @setup_workload, @compile_workload
using Random

island1(λ, φ) = exp(-((λ - 70)^2 + (φ - 55)^2) / 50)
island2(λ, φ) = exp(-((λ - 250)^2 + (φ - 55)^2) / 50)
gaussian_islands(λ, φ) = -1000 + 1100 * (island1(λ, φ) + island2(λ, φ))

step(φ) = (1 - tanh((abs(φ) - 40) / 5)) / 2
Tᵢ(λ, φ, z) = (30 + 1e-3 * z) * step(φ) + rand()
Sᵢ(λ, φ, z) = -5e-3 * z + rand()

function baroclinic_instability_model(grid)
    tracers = (:T, :S)
    equation_of_state = SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)
    buoyancy = SeawaterBuoyancy(; equation_of_state)

    vitd = VerticallyImplicitTimeDiscretization()
    closure = VerticalScalarDiffusivity(vitd, κ=1e-5, ν=1e-4)

    model = HydrostaticFreeSurfaceModel(; grid, tracers, buoyancy, closure,
                                        free_surface = SplitExplicitFreeSurface(substeps=30),
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        momentum_advection = WENOVectorInvariant(order=5),
                                        tracer_advection = WENO(order=5))

    model.clock.last_Δt = 60
    set!(model, T=Tᵢ, S=Sᵢ)

    return model
end

@setup_workload begin
    Oceananigans.defaults.FloatType = Float64
    Nx, Ny, Nz = 64, 32, 4

    @compile_workload begin
        arch = Oceananigans.Architectures.ReactantState()

        # Two grids
        Δz = 1000 / Nz
        z = convert(Vector{FT}, -1000:Δz:0)

        # tripolar_grid = TripolarGrid(arch; size=(Nx, Ny, Nz), halo=(7, 7, 7), z)
        # tripolar_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(gaussian_islands))

        simple_lat_lon_grid = LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), z,
                                                    longitude = (0, 360),
                                                    latitude = (-80, 80))

        grid = simple_lat_lon_grid 
        model = baroclinic_instability_model(grid)
        compiled_first_time_step! = @compile Oceananigans.TimeSteppers.first_time_step!(model, model.clock.last_Δt)
        compiled_time_step! = @compile Oceananigans.TimeSteppers.time_step!(model, model.clock.last_Δt)

        compiled_first_time_step!(model)
        compiled_time_step!(model)
    end

    # Reset float type
    Oceananigans.defaults.FloatType = Float64
end

end # module

