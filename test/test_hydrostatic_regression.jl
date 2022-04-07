include("dependencies_for_runtests.jl")

using Oceananigans.Grids: topology, XRegLatLonGrid, YRegLatLonGrid, ZRegLatLonGrid

function show_hydrostatic_test(grid, free_surface, comp) 

    typeof(grid) <: XRegLatLonGrid ? gx = :regular : gx = :stretched
    typeof(grid) <: YRegLatLonGrid ? gy = :regular : gy = :stretched
    typeof(grid) <: ZRegLatLonGrid ? gz = :regular : gz = :stretched
 
    arch = grid.architecture
    free_surface_str = string(typeof(free_surface).name.wrapper)
    
    strc = "$(comp ? ", metrics are precomputed" : "")"

    testset_str = "Hydrostatic free turbulence regression [$(typeof(arch)), $(topology(grid, 1)) longitude,  ($gx, $gy, $gz) grid, $free_surface_str]" * strc
    info_str    =  "  Testing Hydrostatic free turbulence [$(typeof(arch)), $(topology(grid, 1)) longitude,  ($gx, $gy, $gz) grid, $free_surface_str]" * strc

    return testset_str, info_str
end

function get_fields_from_checkpoint(filename)
    file = jldopen(filename)

    tracers = keys(file["tracers"])
    tracers = Tuple(Symbol(c) for c in tracers)

    velocity_fields = (u = file["velocities/u/data"],
                       v = file["velocities/v/data"],
                       w = file["velocities/w/data"])

    tracer_fields =
        NamedTuple{tracers}(Tuple(file["tracers/$c/data"] for c in tracers))

    current_tendency_velocity_fields = (u = file["timestepper/Gⁿ/u/data"],
                                        v = file["timestepper/Gⁿ/v/data"],
                                        w = file["timestepper/Gⁿ/w/data"])

    current_tendency_tracer_fields =
        NamedTuple{tracers}(Tuple(file["timestepper/Gⁿ/$c/data"] for c in tracers))

    previous_tendency_velocity_fields = (u = file["timestepper/G⁻/u/data"],
                                         v = file["timestepper/G⁻/v/data"],
                                         w = file["timestepper/G⁻/w/data"])

    previous_tendency_tracer_fields =
        NamedTuple{tracers}(Tuple(file["timestepper/G⁻/$c/data"] for c in tracers))

    close(file)

    solution = merge(velocity_fields, tracer_fields)
    Gⁿ = merge(current_tendency_velocity_fields, current_tendency_tracer_fields)
    G⁻ = merge(previous_tendency_velocity_fields, previous_tendency_tracer_fields)

    return solution, Gⁿ, G⁻
end

include("regression_tests/hydrostatic_free_turbulence_regression_test.jl")

@testset "Hydrostatic Regression" begin
    @info "Running hydrostatic regression tests..."

    for arch in archs
        longitudes = [(-180, 180), (-160, 160)]
        latitudes  = [(-60, 60)]
        zs         = [(-90, 0)]
    
        explicit_free_surface = ExplicitFreeSurface(gravitational_acceleration = 1.0)
        implicit_free_surface = ImplicitFreeSurface(gravitational_acceleration = 1.0,
                                                    solver_method = :PreconditionedConjugateGradient,
                                                    reltol = 0, abstol = 1e-15)
        
            for longitude in longitudes, latitudes in latitudes, z in zs, precompute_metrics in (true, false)
                longitude[1] == -180 ? size = (180, 60, 3) : size = (160, 60, 3)
                grid  = LatitudeLongitudeGrid(arch; size, longitude, latitude, z, precompute_metrics, halo=(2, 2, 2))

            for free_surface in [explicit_free_surface, implicit_free_surface]
                                    
                # GPU + ImplicitFreeSurface + precompute metrics cannot be tested on sverdrup at the moment
                # because "uses too much parameter space (maximum 0x1100 bytes)" error 
                if !(precompute_metrics && free_surface isa ImplicitFreeSurface && arch isa GPU) 

                    testset_str, info_str = show_hydrostatic_test(grid, free_surface, comp)
                    
                    @testset "$testset_str" begin
                        @info "$info_str"
                        run_hydrostatic_free_turbulence_regression_test(grid, free_surface)
                    end
                end
            end
        end
    end
end
