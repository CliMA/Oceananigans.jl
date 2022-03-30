include("dependencies_for_runtests.jl")

VALIDATION_DIR = "../validation/"
EXPERIMENTS = ["stratified_couette_flow"]

for exp in EXPERIMENTS
    script_filepath = joinpath(VALIDATION_DIR, exp, exp * ".jl")
    try
        include(script_filepath)
    catch err
        @error sprint(showerror, err)
    end
end

function run_stratified_couette_flow_validation(arch)
    simulate_stratified_couette_flow(Nxy=16, Nz=8, arch=arch, Ri=0.01, Ni=1, end_time=1e-5)
    return true # We're just checking to make sure the script runs with no errors.
end

@testset "Validation" begin
    @info "Testing validation scripts..."

    for arch in archs
        @testset "Stratified Couette flow validation [$(typeof(arch))]" begin
            @info "  Testing stratified Couette flow validation [$(typeof(arch))]"
            @test run_stratified_couette_flow_validation(arch)
        end
    end
end
