include("dependencies_for_runtests.jl")

using Oceananigans.Grids: topology, XRegLatLonGrid, YRegLatLonGrid, ZRegLatLonGrid

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

@testset "Nonhydrostatic Regression" begin
    @info "Running nonhydrostatic regression tests..."

    for arch in archs
        for grid_type in [:regular, :vertically_unstretched]
            @testset "Thermal bubble [$(typeof(arch)), $grid_type grid]" begin
                @info "  Testing thermal bubble regression [$(typeof(arch)), $grid_type grid]"
                run_thermal_bubble_regression_test(arch, grid_type)
            end

            @testset "Rayleigh–Bénard tracer [$(typeof(arch)), $grid_type grid]]" begin
                @info "  Testing Rayleigh–Bénard tracer regression [$(typeof(arch)), $grid_type grid]"
                run_rayleigh_benard_regression_test(arch, grid_type)
            end

            amd_closure = (AnisotropicMinimumDissipation(), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
            smag_closure = (SmagorinskyLilly(C=0.23, Cb=1, Pr=1), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))

            for closure in (amd_closure, smag_closure)
                closurename = string(typeof(first(closure)).name.wrapper)
                @testset "Ocean large eddy simulation [$(typeof(arch)), $closurename, $grid_type grid]" begin
                    @info "  Testing oceanic large eddy simulation regression [$(typeof(arch)), $closurename, $grid_type grid]"
                    run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
                end
            end
        end
    end
end
