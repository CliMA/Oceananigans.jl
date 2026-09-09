include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: topology, XRegularLLG, YRegularLLG, ZRegularLLG
using Oceananigans.Fields: CenterField
using Oceananigans.TurbulenceClosures: LagrangianAveraging

function get_fields_from_checkpoint(filename)
    file = jldopen(filename)

    # Auto-detect format: new Checkpointer saves under "simulation/model/...",
    # legacy reference data has fields at the file root.
    base = haskey(file, "simulation") ? "simulation/model/" : ""

    tracers = keys(file["$(base)tracers"])
    tracers = Tuple(Symbol(c) for c in tracers)

    velocity_fields = (u = file["$(base)velocities/u/data"],
                       v = file["$(base)velocities/v/data"],
                       w = file["$(base)velocities/w/data"])

    tracer_fields =
        NamedTuple{tracers}(Tuple(file["$(base)tracers/$c/data"] for c in tracers))

    current_tendency_velocity_fields = (u = file["$(base)timestepper/Gⁿ/u/data"],
                                        v = file["$(base)timestepper/Gⁿ/v/data"],
                                        w = file["$(base)timestepper/Gⁿ/w/data"])

    current_tendency_tracer_fields =
        NamedTuple{tracers}(Tuple(file["$(base)timestepper/Gⁿ/$c/data"] for c in tracers))

    previous_tendency_velocity_fields = (u = file["$(base)timestepper/G⁻/u/data"],
                                         v = file["$(base)timestepper/G⁻/v/data"],
                                         w = file["$(base)timestepper/G⁻/w/data"])

    previous_tendency_tracer_fields =
        NamedTuple{tracers}(Tuple(file["$(base)timestepper/G⁻/$c/data"] for c in tracers))

    # Closure prognostic fields. Only present in new-format checkpoints.
    # JLD2 fully deserializes these into the original Tuple/NamedTuple
    # structure, so we just return the loaded object as-is.
    closure_fields = haskey(file, "$(base)closure_fields") ? file["$(base)closure_fields"] : nothing

    # Non-hydrostatic pressure — without restoring this, the next step's
    # velocity correction (u -= Δt * ∇p) starts from zero pressure and
    # diverges from the running-state reference at O(Δt) per step.
    pNHS = haskey(file, "$(base)pressures/pNHS/data") ? file["$(base)pressures/pNHS/data"] : nothing

    close(file)

    solution = merge(velocity_fields, tracer_fields)
    Gⁿ = merge(current_tendency_velocity_fields, current_tendency_tracer_fields)
    G⁻ = merge(previous_tendency_velocity_fields, previous_tendency_tracer_fields)

    return solution, Gⁿ, G⁻, closure_fields, pNHS
end

include("regression_tests/thermal_bubble_regression_test.jl")
include("regression_tests/rayleigh_benard_regression_test.jl")
include("regression_tests/ocean_large_eddy_simulation_regression_test.jl")

@testset "Nonhydrostatic Regression" begin
    @info "Running nonhydrostatic regression tests..."

    archs = nonhydrostatic_regression_test_architectures()

    for arch in archs
        A = typeof(arch)

        for grid_type in [:regular, :vertically_unstretched]
            @testset "Rayleigh–Bénard tracer [$A, $grid_type grid]]" begin
                @info "  Testing Rayleigh–Bénard tracer regression [$A, $grid_type grid]"
                run_rayleigh_benard_regression_test(arch, grid_type)
            end

            if !(arch isa Distributed)
                @testset "Thermal bubble [$A, $grid_type grid]" begin
                    @info "  Testing thermal bubble regression [$A, $grid_type grid]"
                    run_thermal_bubble_regression_test(arch, grid_type)
                end

                amd_closure = (AnisotropicMinimumDissipation(C=1/12), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
                smag_closure = (SmagorinskyLilly(C=0.23, Cb=1, Pr=1), ScalarDiffusivity(ν=1.05e-6, κ=1.46e-7))
                dyn_smag_directional = (DynamicSmagorinsky(averaging=(1, 2)),)
                dyn_smag_lagrangian = (DynamicSmagorinsky(averaging=LagrangianAveraging()),)

                for (closurename, closure) in [("AnisotropicMinimumDissipation", amd_closure),
                                               ("SmagorinskyLilly", smag_closure),
                                               ("DirectionalDynamicSmagorinsky", dyn_smag_directional),
                                               ("LagrangianDynamicSmagorinsky", dyn_smag_lagrangian)]
                    @info "  Testing oceanic large eddy simulation regression [$A, $closurename, $grid_type grid]"
                    run_ocean_large_eddy_simulation_regression_test(arch, grid_type, closure)
                end
            end
        end
    end
end
