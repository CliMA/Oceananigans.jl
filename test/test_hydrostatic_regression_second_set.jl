include("dependencies_for_runtests.jl")
include("data_dependencies.jl")

using Oceananigans.Grids: topology, XRegularLLG, YRegularLLG, ZRegularLLG
using Oceananigans.DistributedComputations: synchronized, Distributed

include("regression_tests/hydrostatic_rotation_regression_test.jl")

@testset "Hydrostatic Regression" begin
    @info "Running hydrostatic regression tests..."

    for arch in archs
        @testset "Rotation with shear tests" begin
            z_static  = ExponentialDiscretization(10, -500, 0)
            z_mutable = ExponentialDiscretization(10, -500, 0; mutable=true)

            closures = (nothing, CATKEVerticalDiffusivity())
            timesteppers = (:QuasiAdamsBashforth2, :SplitRungeKutta3)

            for z in (z_static, z_mutable), closure in closures, timestepper in timesteppers

                if arch isa Distributed && closures isa CATKEVerticalDiffusivity
                    arch = synchronized(arch)
                end

                coord_str = z isa MutableVerticalDiscretization ? "Mutable" : "Static"
                closure_str = isnothing(closure) ? "Nothing" : "CATKE"
                timestepper_str = timestepper == :QuasiAdamsBashforth2 ? "AB2" : "RK3"

                testset_str = "Hydrostatic rotation regression [$(arch), $(coord_str) z, $(closure_str) closure, $(timestepper_str) timestepper]"
                info_str    =  "  Testing $testset_str"

                grid = LatitudeLongitudeGrid(arch; size=(150, 150, 10), latitude=(-80, 80), longitude=(0, 360), z, halo=(4, 4, 4))

                @testset "$testset_str" begin
                    @info "$info_str"
                    run_hydrostatic_rotation_regression_test(grid, closure, timestepper)
                end
            end
        end
    end
end
