include("reactant_test_utils.jl")
include("reactant_correctness_utils.jl")

using CUDA
using Random
using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!
using Oceananigans.Solvers: ConjugateGradientPoissonSolver

#####
##### Solver tests comparing vanilla Oceananigans against `ReactantState`, for solvers whose
##### `solve!` is specialized in `OceananigansReactantExt`.
#####
##### Compilation is covered by every `ReactantState` call below, each of which compiles before it
##### runs. Going through `ConjugateGradientPoissonSolver`, the in-tree consumer of
##### `ConjugateGradientSolver`, exercises the real linear operator, preconditioner, residual norm
##### and gauge condition rather than stand-ins written for the test.
#####

const CG_GRID_KW = (size=(4, 4, 4), halo=(3, 3, 3), extent=(1, 1, 1),
                    topology=(Periodic, Periodic, Bounded))

"""
Random CPU arrays shaped for the velocity and pressure fields of `CG_GRID_KW`, shared by the
vanilla and Reactant problems so both start bit-for-bit identical.
"""
function cg_data(seed=1234)
    Random.seed!(seed)
    grid = RectilinearGrid(CPU(); CG_GRID_KW...)
    u, v, w = VelocityFields(grid)
    return (u = randn(size(u)...), v = randn(size(v)...),
            w = randn(size(w)...), p = randn(size(CenterField(grid))...))
end

"""
    cg_pressure(arch, data; kw...)

Project a divergent velocity field on `arch` and return the resulting pressure. `kw...` go to
`ConjugateGradientPoissonSolver`.
"""
function cg_pressure(arch, data; kw...)
    grid = RectilinearGrid(arch; CG_GRID_KW...)

    u, v, w = VelocityFields(grid)
    set!(u, data.u)
    set!(v, data.v)
    set!(w, data.w)
    fill_halo_regions!((u, v, w))
    U = (; u, v, w)

    # Nonzero so that "the solver left `pressure` alone" is a meaningful assertion.
    pressure = CenterField(grid)
    set!(pressure, data.p)

    solver = ConjugateGradientPoissonSolver(grid; kw...)

    # Δt = 1 avoids the pressure rescaling in `solve_for_pressure!`.
    if arch isa ReactantState
        rsolve! = @compile raise=true solve_for_pressure!(pressure, solver, nothing, U, 1)
        rsolve!(pressure, solver, nothing, U, 1)
    else
        solve_for_pressure!(pressure, solver, nothing, U, 1)
    end

    return pressure
end

@testset "Reactant solvers" begin
    @info "Testing Reactant solvers (comparing vanilla Oceananigans vs ReactantState)..."

    vanilla_arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
    data = cg_data()

    @testset "ConjugateGradientSolver" begin

        @testset "converged pressure matches vanilla" begin
            kw = (reltol=1e-12, abstol=1e-12, maxiter=1000)
            @test compare_interior("pressure", cg_pressure(vanilla_arch, data; kw...),
                                               cg_pressure(ReactantState(), data; kw...), rtol=1e-6)
        end

        # With reltol = abstol = 0 the tolerance is zero, so both solvers run exactly `maxiter`
        # iterations regardless of convergence. That pins the traced loop to the eager one step for
        # step and catches an off-by-one in the loop bound.
        @testset "matches vanilla after $maxiter iterations" for maxiter in (1, 5)
            kw = (reltol=0, abstol=0, maxiter=maxiter)
            @test compare_interior("iter$maxiter", cg_pressure(vanilla_arch, data; kw...),
                                                   cg_pressure(ReactantState(), data; kw...), rtol=1e-6)
        end

        # An absolute tolerance far above the initial residual means the problem is already
        # "converged", so neither solver should take a step and the pressure must come back
        # untouched. Without a working tolerance branch the traced loop would run to `maxiter`.
        @testset "returns the initial guess when already converged" begin
            kw = (reltol=0, abstol=1e6, maxiter=1000)
            reactant_pressure = cg_pressure(ReactantState(), data; kw...)
            @test Array(interior(reactant_pressure)) == data.p
            @test compare_interior("untouched", cg_pressure(vanilla_arch, data; kw...),
                                                reactant_pressure)
        end
    end
end
