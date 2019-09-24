@testset "Time steppers" begin
    ϕ = 2
    Δt = 1/5
    @test RK3(x -> x^2, ϕ, Δt) ≈ 20653834//6328125
end

