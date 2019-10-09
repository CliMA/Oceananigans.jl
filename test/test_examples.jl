EXAMPLES_DIR = "../examples/"

function run_example(replace_strings, example_name, module_suffix="")
    example_filepath = joinpath(EXAMPLES_DIR, example_name * ".jl")
    txt = read(example_filepath, String)

    for strs in replace_strings
        txt = replace(txt, strs[1] => strs[2])
    end

    test_script_filepath = example_name * "_example_test.jl"

    open(test_script_filepath, "w") do f
        write(f, "module Test_$example_name" * "_$module_suffix\n")
        write(f, txt)
        write(f, "\nend # module")
    end

    try
        include(test_script_filepath)
    catch err
        @error sprint(showerror, err)
        rm(test_script_filepath)
        return false
    end

    rm(test_script_filepath)
    return true
end

@testset "Examples" begin
    println("Testing examples...")

    for arch in archs
        @testset "Wind and convection mixing example [$(typeof(arch))]" begin
            println("  Testing wind and convection-driving mixing example [$(typeof(arch))]")

            replace_strings = [ ("Nz = 48", "Nz = 16"),
                                ("while model.clock.time < end_time", "while model.clock.iteration < 1"),
                                ("time_step!(model, 10, wizard.Δt)", "time_step!(model, 1, wizard.Δt)"),
                              ]

            arch == GPU() && push!(replace_strings, ("architecture = CPU()", "architecture = GPU()"))

            @test run_example(replace_strings, "ocean_wind_mixing_and_convection", string(typeof(arch)))
            rm("ocean_wind_mixing_and_convection.jld2")
        end
    end

    @testset "Simple diffusion example" begin
        println("  Testing simple diffusion example")

        replace_strings = [ ("N = (1, 1, 128)", "N = (1, 1, 16)"),
                            ("Nt = 1000", "Nt = 2")
                          ]

        @test run_example(replace_strings, "simple_diffusion")
    end

    @testset "Internal wave example" begin
        println("  Testing internal wave example")

        replace_strings = [ ("Nx = 128", "Nx = 16"),
                            ("i = 1:10", "i = 1:1"),
                            ("Nt = 200", "Nt = 2"),
                          ]

        @test run_example(replace_strings, "internal_wave")
    end

    @testset "Two-dimensional turbulence example" begin
        println("  Testing two-dimensional turbulence example")

        replace_strings = [ ("N=(128, 128, 1)", "N=(16, 16, 1)"),
                            ("i = 1:10", "i = 1:1"),
                            ("Nt = 100", "Nt = 2")
                          ]

        @test run_example(replace_strings, "two_dimensional_turbulence")
    end

end
