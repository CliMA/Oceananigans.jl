include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, DiffusiveFormulation, AdvectiveFormulation

function gm_tracer_remains_finite(arch, FT; skew_flux_formulation, horizontal_direction)
    eddy_closure = IsopycnalSkewSymmetricDiffusivity(FT, κ_skew=1e3, κ_symmetric=1e3,
                                                     skew_flux_formulation=skew_flux_formulation)

    Nx = Ny = Nz = 16
    H = 100
    L = 10e3

    z = ExponentialDiscretization(Nz, -H, 0)

    closure = eddy_closure
    buoyancy = BuoyancyTracer()
    tracers = :b

    # Create grid and initial condition based on direction
    if horizontal_direction == :x
        # Slope in x-direction (Flat in y)
        grid = RectilinearGrid(arch, FT;
                               size = (Nx, Nz),
                               x = (0, L),
                               z,
                               topology = (Bounded, Flat, Bounded))

        model = HydrostaticFreeSurfaceModel(grid; buoyancy, closure, tracers)

        set!(model, b = (x, z) -> x / 10000)

    elseif horizontal_direction == :y
        # Slope in y-direction (Flat in x)
        grid = RectilinearGrid(arch, FT;
                               size = (Nx, Nz),
                               y = (0, L),
                               z,
                               topology = (Flat, Bounded, Bounded))

        model = HydrostaticFreeSurfaceModel(grid; buoyancy, closure, tracers)

        set!(model, b = (y, z) -> y / 10000)

    else # :xy - full 3D
        # Slope in both x and y directions
        grid = RectilinearGrid(arch, FT;
                               size = (Nx, Ny, Nz),
                               x = (0, L),
                               y = (0, L),
                               z,
                               topology = (Bounded, Bounded, Bounded))

        model = HydrostaticFreeSurfaceModel(grid; buoyancy, closure, tracers)

        set!(model, b = (x, y, z) -> (x + y) / 10000)
    end

    # Time step the model 10 times
    Δt = convert(FT, 10)
    for _ in 1:20
        time_step!(model, Δt)
    end

    b = model.tracers.b

    # Check that buoyancy field contains no NaN values
    return !any(isnan, b)
end

@testset "Gent-McWilliams with infinite slope" begin
    @info "Testing Gent-McWilliams formulations with infinite slope..."

    formulations = [DiffusiveFormulation(), AdvectiveFormulation()]
    directions = [:x, :y, :xy]

    for arch in archs, FT in float_types, formulation in formulations, direction in directions
        formulation_name = typeof(formulation).name.name
        @testset "GM $formulation_name $direction direction [$arch, $FT]" begin
            @info "  Testing GM $formulation_name $direction direction [$arch, $FT]..."
            @test gm_tracer_remains_finite(arch, FT;
                                           skew_flux_formulation=formulation,
                                           horizontal_direction=direction)
        end
    end
end
