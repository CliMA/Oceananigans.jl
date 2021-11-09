using Oceananigans.Grids: topology
using CUDA

include("utils_for_runtests.jl")
include("data_dependencies.jl")

archs = test_architectures()


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

include("regression_tests/thermal_bubble_regression_test.jl")
include("regression_tests/rayleigh_benard_regression_test.jl")
include("regression_tests/ocean_large_eddy_simulation_regression_test.jl")
include("regression_tests/hydrostatic_free_turbulence_regression_test.jl")

@testset "Regression" begin
    @info "Running regression tests..."

    for arch in archs
        # for grid_type in [:regular, :vertically_unstretched]
        #     @testset "Thermal bubble [$(typeof(arch)), $grid_type grid]" begin
        #         @info "  Testing thermal bubble regression [$(typeof(arch)), $grid_type grid]"
        #         run_thermal_bubble_regression_test(arch, grid_type)
        #     end

        #     @testset "Rayleigh–Bénard tracer [$(typeof(arch)), $grid_type grid]]" begin
        #         @info "  Testing Rayleigh–Bénard tracer regression [$(typeof(arch)), $grid_type grid]"
        #         run_rayleigh_benard_regression_test(arch, grid_type)
        #     end

        #     for closure in (AnisotropicMinimumDissipation(ν=1.05e-6, κ=1.46e-7), SmagorinskyLilly(C=0.23, Cb=1, Pr=1, ν=1.05e-6, κ=1.46e-7))
        #         closurename = string(typeof(closure).name.wrapper)
        #         @testset "Ocean large eddy simulation [$(typeof(arch)), $closurename, $grid_type grid]" begin
        #             @info "  Testing oceanic large eddy simulation regression [$(typeof(arch)), $closurename, $grid_type grid]"
        #             run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
        #         end
        #     end
        # end

        # Hydrostatic regression test

        longitude = ((-180, 180), collect(-180:2:180)) #, (-160, 160), collect(-160:2:160))
        latitude  = ((-60, 60), collect(-60:2:60))
        zcoord    = ((-90, 0), collect(-90:30:0))

        explicit_free_surface = ExplicitFreeSurface(gravitational_acceleration=1.0)
        implicit_free_surface = ImplicitFreeSurface(gravitational_acceleration = 1.0,
                                                   solver_method = :PreconditionedConjugateGradient,
                                                   tolerance = 1e-15)

        for lon in longitude, lat in latitude, z in zcoord, comp in (false)

            lon[1] == -180 ? N = (180, 60, 3) : N = (160, 60, 3)

            grid  = LatitudeLongitudeGrid(size = N,
                                     longitude = lon,
                                      latitude = lat,
                                             z = z,
                                          halo = (2, 2, 2),
                                  architecture = arch,
                            precompute_metrics = comp)

            for free_surface in [explicit_free_surface, implicit_free_surface]
                                 
                # GPU + ImplicitFreeSurface + precompute metrics is not compatible at the moment. 
                # kernel "uses too much parameter space  (maximum 0x1100 bytes) " error 
                if !(comp && free_surface isa ImplicitFreeSurface && arch isa GPU) 

                    testset_str, info_str = show_hydrostatic_test(grid, free_surface, comp)
                    @testset "$testset_str" begin
                        @info "$info_str"
                        run_hydrostatic_free_turbulence_regression_test(grid, free_surface, arch)
                    end
                end
            end
	    end   
	end
end
