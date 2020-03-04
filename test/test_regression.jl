function summarize_regression_test(field_names, fields, correct_fields)
    for (field_name, φ, φ_c) in zip(field_names, fields, correct_fields)
        Δ = Array(φ) .- φ_c

        Δ_min      = minimum(Δ)
        Δ_max      = maximum(Δ)
        Δ_mean     = mean(Δ)
        Δ_abs_mean = mean(abs, Δ)
        Δ_std      = std(Δ)

        @info(@sprintf("Δ%s: min=%.6g, max=%.6g, mean=%.6g, absmean=%.6g, std=%.6g",
                       field_name, Δ_min, Δ_max, Δ_mean, Δ_abs_mean, Δ_std))
    end
end

interior(a, grid) = view(a, grid.Hx+1:grid.Nx+grid.Hx,
                            grid.Hy+1:grid.Ny+grid.Hy,
                            grid.Hz+1:grid.Nz+grid.Hz)

function get_fields_from_checkpoint(filename)
    file = jldopen(filename)

    tracers = keys(file["tracers"])
    tracers = Tuple(Symbol(c) for c in tracers)

    velocity_fields = (
                       u = file["velocities/u/data"],
                       v = file["velocities/v/data"],
                       w = file["velocities/w/data"],
                      )

    tracer_fields = NamedTuple{tracers}(Tuple(file["tracers/$c/data"] for c in tracers))

    current_tendency_velocity_fields = (
                                        u = file["timestepper/Gⁿ/u/data"],
                                        v = file["timestepper/Gⁿ/v/data"],
                                        w = file["timestepper/Gⁿ/w/data"],
                                       )

    current_tendency_tracer_fields = NamedTuple{tracers}(Tuple(file["timestepper/Gⁿ/$c/data"] for c in tracers))

    previous_tendency_velocity_fields = (
                                         u = file["timestepper/G⁻/u/data"],
                                         v = file["timestepper/G⁻/v/data"],
                                         w = file["timestepper/G⁻/w/data"],
                                        )

    previous_tendency_tracer_fields = NamedTuple{tracers}(Tuple(file["timestepper/G⁻/$c/data"] for c in tracers))

    close(file)

    solution = merge(velocity_fields, tracer_fields)
    Gⁿ = merge(current_tendency_velocity_fields, current_tendency_tracer_fields)
    G⁻ = merge(previous_tendency_velocity_fields, previous_tendency_tracer_fields)

    return solution, Gⁿ, G⁻
end

include("regression_tests/thermal_bubble_regression_test.jl")
include("regression_tests/rayleigh_benard_regression_test.jl")
include("regression_tests/ocean_large_eddy_simulation_regression_test.jl")

@testset "Regression" begin
    @info "Running regression tests..."

    for arch in archs
        @testset "Thermal bubble [$(typeof(arch))]" begin
            @info "  Testing thermal bubble regression [$(typeof(arch))]"
            run_thermal_bubble_regression_test(arch)
        end

        @testset "Rayleigh–Bénard tracer [$(typeof(arch))]" begin
            @info "  Testing Rayleigh–Bénard tracer regression [$(typeof(arch))]"
            run_rayleigh_benard_regression_test(arch)
        end

        @testset "Ocean large eddy simulation [$(typeof(arch))]" begin
            for closure in (AnisotropicMinimumDissipation(), ConstantSmagorinsky())
                closurename = string(typeof(closure).name.wrapper)
                @info "  Testing oceanic large eddy simulation regression [$closurename, $(typeof(arch))]"
                run_ocean_large_eddy_simulation_regression_test(arch, closure)
            end
        end
    end
end
