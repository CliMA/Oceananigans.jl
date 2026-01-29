include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using Random
using CUDA
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid, OrthogonalSphericalShellGrid
using SeawaterPolynomials: TEOS10EquationOfState

loc = (Center, Center, Center)
topo = (Periodic, Periodic, Bounded)
raise = true

# Get vanilla architecture from TEST_ARCHITECTURE env var (set by reactant_test_utils.jl)
vanilla_arch = Distributed(get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU())
reactant_arch = Distributed(ReactantState())

# Test RectilinearGrid with all topologies
kw = (size=(8, 4, 2), halo=(1, 1, 1), extent=(1, 1, 1), topology=topo)
vanilla_grid = RectilinearGrid(vanilla_arch; kw...)
reactant_grid = RectilinearGrid(reactant_arch; kw...)

LX, LY, LZ = loc
vanilla_field = Field{LX, LY, LZ}(vanilla_grid)
reactant_field = Field{LX, LY, LZ}(reactant_grid)

Random.seed!(12345)
data = randn(size(vanilla_field)...)
set!(vanilla_field, data)
set!(reactant_field, data)

fill_halo_regions!(vanilla_field)
# @jit raise=raise fill_halo_regions!(reactant_field)
@jit fill_halo_regions!(reactant_field)

@test compare_parent("halo", vanilla_field, reactant_field)
