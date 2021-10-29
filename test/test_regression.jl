function summarize_regression_test(fields, correct_fields)
    for (field_name, φ, φ_c) in zip(keys(fields), fields, correct_fields)
        Δ = φ .- φ_c

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        matching    = sum(φ .≈ φ_c)
        grid_points = length(φ_c)

        @info @sprintf("Δ%s: min=%+.6e, max=%+.6e, mean=%+.6e, absmean=%+.6e, std=%+.6e (%d/%d matching grid points)",
                       field_name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std, matching, grid_points)
    end
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
include("regression_tests/lat_lon_free_turbulence_regression.jl")

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
        grid_conf  = [:regular, :unstretched]
        precompute = (true, false)
        for grid_x in grid_conf, grid_y in grid_conf, grid_z in grid_conf, compute in precompute
            @testset "Latitude Longitude free turbulence regression [$(typeof(arch)), ($grid_x, $grid_y, $grid_z) grid$(compute ? ", metrics are precomputed" : "")]" begin
                @info "  Testing Latitude Longitude free turbulence [$(typeof(arch)), ($grid_x, $grid_y, $grid_z) grid$(compute ? ", metrics are precomputed" : "")]"
                run_lat_lon_free_turbulence_regression_test(grid_x, grid_y, grid_z, arch, compute)
            end
        end
    end
end

