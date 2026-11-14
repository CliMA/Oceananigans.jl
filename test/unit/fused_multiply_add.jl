include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: apply_fused_multiply_add_choice, replace_muladd

contains_value(x, v) = x === v || x isa Expr && any(arg -> contains_value(arg, v), x.args)

@testset "Fused multiply-add" begin
    @testset "replace_muladd" begin
        muladd_macro = GlobalRef(Oceananigans.Advection.MuladdMacro, Symbol("@muladd"))
        expression = macroexpand(@__MODULE__, Expr(:macrocall, muladd_macro, LineNumberNode(0), :(x * y + z)))
        replaced = replace_muladd(expression, :replacement)

        @test contains_value(expression, muladd)
        @test contains_value(replaced, :replacement)
        @test !contains_value(replaced, muladd)
    end

    @testset "apply_fused_multiply_add_choice" begin
        expression = :(muladd(x, y, z))

        @test apply_fused_multiply_add_choice(expression, "muladd") === expression
        @test !contains_value(apply_fused_multiply_add_choice(expression, "fma"), :muladd)
        @test contains_value(apply_fused_multiply_add_choice(expression, "fma"), GlobalRef(Base, :fma))
        @test_throws ErrorException apply_fused_multiply_add_choice(expression, "bogus")
    end
end
