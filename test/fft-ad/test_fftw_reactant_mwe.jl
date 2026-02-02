#=
MWE: FFTW incompatibility with Reactant arrays
Demonstrates why OceananigansReactantExt/Solvers.jl is needed.
=#

using FFTW
using Reactant
using Test

Reactant.set_default_backend("cpu")

@testset "FFTW + Reactant MWE" begin

    @testset "FFTW works with regular Array" begin
        A = zeros(ComplexF64, 4, 4, 4)
        plan = FFTW.plan_fft!(A, [1, 2, 3])
        @test plan !== nothing
    end

    @testset "FFTW fails with ConcreteRArray - no pointer" begin
        A_julia = zeros(ComplexF64, 4, 4, 4)
        A_reactant = Reactant.to_rarray(A_julia)
        
        # This fails because Reactant arrays don't expose raw pointers
        @test_throws ErrorException FFTW.plan_fft!(A_reactant, [1, 2, 3])
    end

    @testset "AbstractFFTs works with Reactant during tracing" begin
        using AbstractFFTs
        
        A_julia = zeros(ComplexF64, 4, 4, 4)
        A_reactant = Reactant.to_rarray(A_julia)
        
        # This works because Reactant intercepts AbstractFFTs and compiles to XLA
        function do_fft(A)
            return AbstractFFTs.fft(A)
        end
        
        compiled = Reactant.@compile do_fft(A_reactant)
        @test compiled !== nothing
        
        result = compiled(A_reactant)
        @test size(result) == (4, 4, 4)
    end

end
