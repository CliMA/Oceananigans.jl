include("dependencies_for_runtests.jl")


# Generate some random points in a single binade [1;2) interval
function test_data_in_single_binade(::Type{FT}, size) where {FT}
    prng = Random.Xoshiro(44)
    return rand(prng, FT, size) .+ 1.0
end

@testset "CPU newton_div" for (FT, WCT) in Iterators.product((Float32, Float64),
                                                            (Oceananigans.Utils.NormalDivision,
                                                             Oceananigans.Utils.ConvertingDivision{Float32}))
    test_input = test_data_in_single_binade(FT, 1024)

    ref = similar(test_input)
    output = similar(test_input)

    ref .= FT(π) ./ test_input
    output .= Oceananigans.Utils.newton_div.(WCT, FT(π), test_input)

    @test isapprox(ref, output)
end


function append_weight_computation_type!(list, weno::WENO{<:Any, <:Any, WCT}) where {WCT}
    push!(list, WCT)
    append_weight_computation_type!(list, weno.buffer_scheme)
end
append_weight_computation_type!(::Any, ::Any) = nothing

# Extract all weight computation types from WENO
# Assumes a non-weno buffer scheme will not have WENO buffer scheme
function get_weight_computation_from_weno_advection(weno::WENO)
    weight_computation_types = DataType[]
    append_weight_computation_type!(weight_computation_types, weno)
    return weight_computation_types
end

@testset "Verify WENO schemes construction" begin

    # WENO
    weno5 = WENO(order=7; weight_computation=Oceananigans.Utils.NormalDivision)
    weight_computation_types = get_weight_computation_from_weno_advection(weno5)
    @test all(weight_computation_types .== Oceananigans.Utils.NormalDivision)

    # Vector Invariant WENO
    vector_weno = WENOVectorInvariant(order=9, weight_computation=Oceananigans.Utils.BackendOptimizedDivision)

    for field_name in fieldnames(typeof(vector_weno))
        field = getfield(vector_weno, field_name)
        if field isa WENO
            weight_computation_types = get_weight_computation_from_weno_advection(field)
            @test all(weight_computation_types .== Oceananigans.Utils.BackendOptimizedDivision)
        end
    end
end
