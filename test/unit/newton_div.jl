include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using CUDA

# Generate some random points in a single binade [1;2) interval
function test_data_in_single_binade(::Type{FT}, size) where {FT}
    prng = Random.Xoshiro(44)
    return rand(prng, FT, size) .+ 1
end

@testset "CPU newton_div: $FT $WCT" for (FT, WCT) in Iterators.product((Float32, Float64),
                                                            (Oceananigans.Utils.NormalDivision,
                                                             Oceananigans.Utils.ConvertingDivision{Float32}))
    test_input = test_data_in_single_binade(FT, 1024)

    ref = similar(test_input)
    output = similar(test_input)

    ref .= FT(π) ./ test_input
    output .= Oceananigans.Utils.newton_div.(WCT, FT(π), test_input)

    @test isapprox(ref, output)
end

if CUDA.functional()

    @testset "CUDA newton_div: $FT" for FT in (Float32, Float64)
        test_input = CuArray(test_data_in_single_binade(FT, 1024))

        WCT = Oceananigans.Utils.BackendOptimizedDivision

        ref = similar(test_input)
        output = similar(test_input)

        ref .= FT(π) ./ test_input
        output .= Oceananigans.Utils.newton_div.(WCT, FT(π), test_input)

        @test isapprox(ref, output)
    end
end

weight_computation(::WENO{<:Any, <:Any, WCT}) where WCT = WCT

@testset "Verify WENO schemes construction" begin

    # WENO
    weno7 = WENO(order=7; weight_computation=Oceananigans.Utils.NormalDivision)
    @test weight_computation(weno7) == Oceananigans.Utils.NormalDivision

    # Vector Invariant WENO
    vector_weno = WENOVectorInvariant(order=9, weight_computation=Oceananigans.Utils.BackendOptimizedDivision)

    for field_name in fieldnames(typeof(vector_weno))
        field = getfield(vector_weno, field_name)
        if field isa WENO
            @test weight_computation(field) == Oceananigans.Utils.BackendOptimizedDivision
        end
    end
end
