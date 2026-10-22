include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: fully_supported_float_types
using Oceananigans.Advection: fused_multiply_add, replace_muladd

BFloat16 = last(fully_supported_float_types)

contains_value(x, v) = x === v || x isa Expr && any(arg -> contains_value(arg, v), x.args)

@testset "Fused multiply-add" begin
    @testset "fused_multiply_add rounds once [$FT]" for FT in (Float32, Float64)
        a, b = FT(1) + eps(FT), FT(1) - eps(FT)
        c = -FT(1)

        @test fused_multiply_add(a, b, c) == fma(a, b, c)
        @test fused_multiply_add(a, b, c) != a * b + c
    end

    @testset "fused_multiply_add accepts mixed and BFloat16 arguments" begin
        @test fused_multiply_add(2, 3, 1) == 7
        @test fused_multiply_add(2, 3.0, 1) === 7.0
        @test fused_multiply_add(BFloat16(2), BFloat16(3), BFloat16(1)) === BFloat16(7)
        @test fused_multiply_add(2, BFloat16(3), BFloat16(1)) === BFloat16(7)
    end

    @testset "replace_muladd" begin
        muladd_macro = GlobalRef(Oceananigans.Advection.MuladdMacro, Symbol("@muladd"))
        expression = macroexpand(@__MODULE__, Expr(:macrocall, muladd_macro, LineNumberNode(0), :(x * y + z)))
        replaced = replace_muladd(expression, :replacement)

        @test contains_value(expression, muladd)
        @test contains_value(replaced, :replacement)
        @test !contains_value(replaced, muladd)
    end
end
