function print_interpolant(k, r)
    cs = coefficients(k, r)

    ssign(n) = n >= 0 ? "+" : "-"
    ssubscript(n) = n == 0 ? "i"  : "i" * ssign(n) * string(abs(n))

    print("u(i+1//2) = ")
    for (j, c) in enumerate(cs)
        c_s = ssign(c) * " " * string(abs(c))
        ss_s = ssubscript(j-r-1)
        print(c_s * " u(" * ss_s * ") ")
    end
    print("\n")
end

function print_interpolants(k)
    for r in -1:k-1
        print_interpolant(k, r)
    end
    Γ = optimal_weights(k)
    println("Optimal weights γᵣ: $Γ")
end

# Recreating entries from Table 2.1 of Shu (1998) lecture notes.

println("WENO-5 [Compare with Table 2.1 of Shu (1998) and equation (2.15) from Shu (2009)]:")
print_interpolants(3)

println("\nWENO-7 [Compare with Table 2.1 of Shu (1998)]:")
print_interpolants(4)

println("\nWENO-9 [Compare with Table 2.1 of Shu (1998) and equation (2.14) of Shu (2009)]:")
print_interpolants(5)

@testset "WENO reconstruction" begin
    @testset "WENO reconstruction weights" begin
        
        # Compare with Table 2.1 of Shu (1998).

        @testset "WENO-5" begin
            @test coefficients(3, -1) == [11//6, -7//6,  1//3]
            @test coefficients(3,  0) == [ 1//3,  5//6, -1//6]
            @test coefficients(3,  1) == [-1//6,  5//6,  1//3]
            @test coefficients(3,  2) == [ 1//3, -7//6, 11//6]
        end

        @testset "WENO-7" begin
            @test coefficients(5, -1) == [ 137//60, -163//60, 137//60,  -21//20,   1//5]
            @test coefficients(5,  0) == [   1//5,    77//60, -43//60,   17//60,  -1//20]
            @test coefficients(5,  1) == [  -1//20,    9//20,  47//60,  -13//60,   1//30]
            @test coefficients(5,  2) == [   1//30,  -13//60,  47//60,    9//20,  -1//20]
            @test coefficients(5,  3) == [  -1//20,   17//60, -43//60,   77//60,   1//5]
            @test coefficients(5,  4) == [   1//5,   -21//20, 137//60, -163//60, 137//60]
        end
    end

    @testset "WENO optimal weights" begin
        @testset "WENO-5" begin
            @test optimal_weights(3) == [1//10, 3//5, 3//10]
        end

        @testset "WENO-7" begin
            @test optimal_weights(5) == [1//126, 10//63, 10//21, 20//63, 5//126]
        end
    end
end

