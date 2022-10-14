using Oceananigans
using Oceananigans.BoundaryConditions
using SeawaterPolynomials

grid = RectilinearGrid(size = 100, z = (-1000, 0), topology = (Flat, Flat, Bounded))  

T = CenterField(grid)
S = CenterField(grid)

set!(T, (x, y, z) -> sin(z / 1000 * 2π) * 10 + 10)
set!(S, (x, y, z) -> 35 + cos(z / 1000 * 2π))

fill_halo_regions((T, S))
eos = SeawaterPolynomials.TEOS10.TEOS10EquationOfState(reference_density = 1029.0)

bL = SeawaterBuoyancy(equation_of_state = LinearEquationOfState())
bT = SeawaterBuoyancy(equation_of_state = eos)

C = (; T, S)
using Oceananigans.BuoyancyModels: ∂z_b, buoyancy_perturbation

bzL_op = KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, computed_dependencies = (bL, C))
bzT_op = KernelFunctionOperation{Center, Center, Face}(∂z_b, grid, computed_dependencies = (bT, C))

bL_op = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation, grid, computed_dependencies = (bL, C))
bT_op = KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbation, grid, computed_dependencies = (bT, C))


bL = compute!(Field(bL_op))
bT = compute!(Field(bT_op))

∂zbL = compute!(Field(bzL_op))
∂zbT = compute!(Field(bzT_op))

∂zbL2 = compute!(Field(∂z(bL_op)))
∂zbT2 = compute!(Field(∂z(bT_op)))

z = znodes(Face, grid)[2:end-1]

fig = Figure()
ax = Axis(fig[1, 1])
lines!(interior(∂zbL)[1, 1, 2:end-1], z)
lines!(interior(∂zbT)[1, 1, 2:end-1], z)
lines!(interior(∂zbL2)[1, 1, 2:end-1], z)
lines!(interior(∂zbT2)[1, 1, 2:end-1], z)
