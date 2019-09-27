EXAMPLES_DIR = "../examples/"

function run_deepening_mixed_layer_example(arch)
    example_filepath = joinpath(EXAMPLES_DIR, "deepening_mixed_layer.jl")

    txt = read(example_filepath, String)

    arch == GPU() && (txt = replace(txt, "arch = CPU()" => "arch = GPU()"))

    txt = replace(txt, "N = 32" => "N = 16")
    txt = replace(txt, "model.clock.time < tf" => "model.clock.time < 0.5")
    txt = replace(txt, "time_step!(model, 10, wizard.Δt)" => "time_step!(model, 1, wizard.Δt)")

    test_script_filepath = "deepening_mixed_layer_example_cpu_test.jl"
    open(test_script_filepath, "w") do f
        write(f, txt)
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

function run_example(replace_strings, example_name)
    example_filepath = joinpath(EXAMPLES_DIR, example_name * ".jl")
    txt = read(example_filepath, String)

    for strs in replace_strings
        txt = replace(txt, strs[1] => strs[2])
    end

    test_script_filepath = example_name * "_example_test.jl"

    open(test_script_filepath, "w") do f
        write(f, "module Test_$example_name\n")
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

    #=
    for arch in archs
        @testset "Deepening mixed layer example [$(typeof(arch))]" begin
            println("  Testing deepening mixed layer example [$(typeof(arch))]")
            @test run_deepening_mixed_layer_example(arch)
        end
    end
    =#

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
                            ("Nt = 200", "Nt = 2")
                          ]

        @test run_example(replace_strings, "internal_wave")
    end

    @testset "Two-dimensional turbulence example" begin
        println("  Testing two-dimensional turbulence example")

        replace_strings = [ ("N = (128, 128, 1)", "N = (16, 16, 1)"),
                            ("i = 1:10", "i = 1:1"),
                            ("Nt = 100", "Nt = 2")
                          ]

        @test run_example(replace_strings, "two_dimensional_turbulence")
    end



end
