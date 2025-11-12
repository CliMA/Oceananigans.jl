include("dependencies_for_runtests.jl")

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, DiffusiveFormulation, AdvectiveFormulation

function gm_tracer_remains_finite(arch, FT; skew_flux_formulation, horizontal_direction)
    eddy_closure = IsopycnalSkewSymmetricDiffusivity(FT, κ_skew=1e3, κ_symmetric=1e3, 
                                                     skew_flux_formulation=skew_flux_formulation)
    
    nx = 16
    ny = 16
    nz = 16

    z_faces = ExponentialDiscretization(nz, -1, 0)
    
    # Create grid and initial condition based on direction
    if horizontal_direction == :x
        # Slope in x-direction (Flat in y)
        grid = RectilinearGrid(arch, FT;
                               size = (nx, nz),
                               x = (0, 1),
                               z = z_faces,
                               topology = (Bounded, Flat, Bounded))
        
        model = HydrostaticFreeSurfaceModel(; grid,
                                            buoyancy = BuoyancyTracer(),
                                            closure = eddy_closure,
                                            tracers = :b)
        
        set!(model, b = (x, z) -> x / 10000)
        
    elseif horizontal_direction == :y
        # Slope in y-direction (Flat in x)
        grid = RectilinearGrid(arch, FT;
                               size = (ny, nz),
                               y = (0, 1),
                               z = z_faces,
                               topology = (Flat, Bounded, Bounded))
        
        model = HydrostaticFreeSurfaceModel(; grid,
                                            buoyancy = BuoyancyTracer(),
                                            closure = eddy_closure,
                                            tracers = :b)
        
        set!(model, b = (y, z) -> y / 10000)
        
    else # :xy - full 3D
        # Slope in both x and y directions
        grid = RectilinearGrid(arch, FT;
                               size = (nx, ny, nz),
                               x = (0, 1),
                               y = (0, 1),
                               z = z_faces,
                               topology = (Bounded, Bounded, Bounded))
        
        model = HydrostaticFreeSurfaceModel(; grid,
                                            buoyancy = BuoyancyTracer(),
                                            closure = eddy_closure,
                                            tracers = :b)
        
        set!(model, b = (x, y, z) -> (x + y) / 10000)
    end
    
    # Time step the model 10 times
    Δt = convert(FT, 1)
    for n in 1:10
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
