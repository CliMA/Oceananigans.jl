function run_deepening_mixed_layer_example(arch)
    txt = read("../examples/deepening_mixed_layer.jl", String)

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
    catch e
        @error e
        rm(test_script_filepath)
        return false
    end

    rm(test_script_filepath)
    return true
end

@testset "Examples" begin
    println("Testing examples...")

    for arch in archs
        @testset "Deepening mixed layer example [$(typeof(arch))]" begin
            println("  Testing deepening mixed layer example [$(typeof(arch))]")
            @test run_deepening_mixed_layer_example(arch)
        end
    end
end
