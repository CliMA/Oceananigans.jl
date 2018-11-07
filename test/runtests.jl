using
  Seapickle,
  Test

function testeos()
  T0 = 283
  S0 = 35
  p0 = 1e5
  rho0 = 1.027e3
  ρ(T0, S0, p0) == rho0
end

@testset "Equation of State" begin
  @test testeos()
end
