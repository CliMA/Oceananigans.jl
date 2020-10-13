example_filepath(example_name, examples_dir="../examples/") =
    joinpath(examples_dir, example_name * ".jl")

@testset "Examples" begin
    @info "Testing examples..."

    @testset "One-dimensional diffusion" begin
        @info "  Testing one-dimensional diffusion example"

        replace_strings = [
            ("size = (1, 1, 128)", "size = (1, 1, 1)"),
            ("stop_iteration = 1000", "stop_iteration = 1"),
            ("simulation.stop_iteration += 10000", "simulation.stop_iteration += 100"),
            ("mp4(", "# mp4(")
        ]

        @test run_script(replace_strings, "one_dimensional_diffusion", example_filepath("one_dimensional_diffusion"))
    end

    @testset "Two-dimensional turbulence example" begin
        @info "  Testing two-dimensional turbulence example"

        replace_strings = [
            ("size=(128, 128, 1)", "size=(1, 1, 1)"),
            ("stop_iteration=2000", "stop_iteration=1"),
            ("iteration_interval = 20", "iteration_interval = 1"),
            ("mp4(", "# mp4(")
        ]

        @test run_script(replace_strings, "two_dimensional_turbulence", example_filepath("two_dimensional_turbulence"))
    end

    for arch in archs
        @testset "Wind and convection-driven mixing example [$(typeof(arch))]" begin
            @info "  Testing wind and convection-driven mixing example [$(typeof(arch))]"

            replace_strings = [
                ("Nz = 32", "Nz = 1"),
                ("iteration_interval=10", "iteration_interval=1"),
                ("for i in 1:100", "for i in 1:1"),
                ("stop_iteration += 10", "stop_iteration += 1"),
                ("mp4(", "# mp4(")
            ]

            if arch isa GPU
                push!(replace_strings, ("architecture = CPU()", "architecture = GPU()"))
            end

            @test run_script(replace_strings, "ocean_wind_mixing_and_convection",
                             example_filepath("ocean_wind_mixing_and_convection"),
                             string(typeof(arch)))

            rm("ocean_wind_mixing_and_convection.jld2", force=true)
        end
    end

    @testset "Ocean convection with plankton example" begin
        @info "  Testing ocean convection with plankton example"

        replace_strings = [
            ("Nz = 128", "Nz = 1"),
            ("iteration_interval=100", "iteration_interval=1"),
            ("for i = 1:100", "for i = 1:1"),
            ("stop_iteration += 100", "stop_iteration += 1"),
            ("mp4(", "# mp4(")
        ]

        @test run_script(replace_strings, "ocean_convection_with_plankton",
                         example_filepath("ocean_convection_with_plankton"))
    end

    @testset "Internal wave example" begin
        @info "  Testing internal wave example"

        replace_strings = [
                           ("size=(128, 1, 128)", "size=(1, 1, 1)"),
                           ("stop_iteration = 100", "stop_iteration = 1"),
                           ("mp4(", "# mp4(")
                          ]

        @test run_script(replace_strings, "internal_wave", example_filepath("internal_wave"))
    end

    @testset "Eady turbulence" begin
        @info "  Testing Eady turbulence example"

        replace_strings = [
                           ("size=(48, 48, 16)", "size=(1, 1, 1)"),
                           ("stop_time = 10day", "stop_time = 600"),
                          ]

        @test run_script(replace_strings, "eady_turbulence", example_filepath("eady_turbulence"))
    end
end
