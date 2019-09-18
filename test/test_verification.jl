VERIFICATION_DIR = "../verification/"

function run_stratified_couette_flow_verification(arch)
    script_filepath = joinpath(VERIFICATION_DIR, "stratified_couette_flow", "stratified_couette_flow.jl")

    try
        include(script_filepath)
        simulate_stratified_couette_flow(Nxy=16, Nz=8, arch=CPU(), Ri=0.01, end_time=1e-15)
    catch e
        @error e
        return false
    end
    
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
