using
  OceanDispatch,
  Test

# function testeos()
#   T0 = 283
#   S0 = 35
#   p0 = 1e5
#   rho0 = 1.027e3
#   ρ(T0, S0, p0) == rho0
# end
#
# @testset "Equation of State" begin
#   @test testeos()
# end

@testset "Grid" begin
  include("test_grid.jl")
  @test test_grid_size()
  @test test_Δx()
  @test test_Δy()
  @test test_Δz()
  @test test_cell_volume()
end

@testset "Fields" begin
  include("test_field.jl")
  @test test_initfield()
  @test test_setfield()
  @test test_addfield()
end
