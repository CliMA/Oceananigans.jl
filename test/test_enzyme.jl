using Oceananigans
using Enzyme

# Required presently
Enzyme.API.runtimeActivity!(true)

EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true

f(grid) = CenterField(grid)

@testset "Enzyme Unit Tests" begin
	arch=CPU()
	FT=Float64

	N = 100
	topo = (Periodic, Flat, Flat)
	grid = RectilinearGrid(arch, FT, topology=topo, size=(N), halo=2, x=(-1, 1), y=(-1, 1), z=(-1, 1))
	fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, typeof(Const(grid)))

	tape, primal, shadow = fwd(Const(f), Const(grid) )

	@show tape, primal, shadow

	@test size(primal) == size(shadow)
end