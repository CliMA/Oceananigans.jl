VERIFICATION_DIR = "../verification/"
EXPERIMENTS = ["stratified_couette_flow"]

for exp in EXPERIMENTS
    script_filepath = joinpath(VERIFICATION_DIR, exp, exp * ".jl")
    try
        include(script_filepath)
    catch err
        @error sprint(showerror, err)
    end
end

function run_stratified_couette_flow_verification(arch)
    simulate_stratified_couette_flow(Nxy=16, Nz=8, arch=arch, Ri=0.01, end_time=1e-15)
    return true
end


@testset "Verification" begin
    println("Testing verification scripts...")

    for arch in archs
        @testset "Stratified Couette flow verification [$(typeof(arch))]" begin
            println("  Testing stratified Couette flow verification [$(typeof(arch))]")
            @test run_stratified_couette_flow_verification(arch)
        end
    end
end
