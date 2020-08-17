using Printf
using TimerOutputs
using OrderedCollections

using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.AbstractOperations

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

FT = Float64
Nt = 10  # Number of iterations to use for benchmarking time stepping.

         archs = [CPU()]        # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()] # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Run benchmarks
#####

for arch in archs
    N = arch isa CPU ? (32, 32, 32) : (256, 256, 256)

    grid = RegularCartesianGrid(size=N, extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid,
                                buoyancy=nothing, tracers=(:α, :β, :γ, :δ, :ζ))

    ε(x, y, z) = randn()
    ε⁺(x, y, z) = abs(randn())
    set!(model, u=ε, v=ε, w=ε, α=ε, β=ε, γ=ε, δ=ε, ζ=ε⁺)

    u, v, w = model.velocities
    α, β, γ, δ, ζ = model.tracers

    dump_field = Field(Cell, Cell, Cell, arch, grid)

    test_cases = OrderedDict(
        "-α"      => -α,
        "√ζ"      => √ζ,
        "sin(β)"  => sin(β),
        "cos(γ)"  => cos(γ),
        "exp(δ)"  => exp(δ),
        "tanh(ζ)" => tanh(ζ),
        "α + β"             => α - β,
        "α + β - γ"         => α + β - γ,
        "α * β * γ * δ"     => α * β * γ * δ,
        "α * β - γ * δ / ζ" => α * β - γ * δ / ζ,
        "u^2 + v^2" => u^2 + v^2,
        "(u^2 + v^2 + w^2) / 2" => (u^2 + v^2 + w^2) / 2,
        "∂x(α)" => ∂x(α),
        "∂y(∂y(β))" => ∂y(∂y(β)),
        "∂z(∂z(∂z(∂z(γ))))" => ∂z(∂z(∂z(∂z(γ)))),
        "∂x(δ + ζ)" => ∂x(δ + ζ),
        "∂x(v) - δy(u)" => ∂x(v) - ∂y(u),
        "∂z(α * β + γ)" => ∂z(α * β + γ),
        "∂x(u) * ∂y(v) + ∂z(w)" => ∂x(u) * ∂y(v) + ∂z(w),
        "∂x(α)^2 + ∂y(α)^2 + ∂z(α)^2" => ∂x(α) + ∂y(α) + ∂z(α)^2,
        "∂x(ζ)^4 + ∂y(ζ)^4 + ∂z(ζ)^4 + 2*∂x(∂x(∂y(∂y(ζ)))) + 2*∂x(∂x(∂z(∂z(ζ)))) + 2*∂y(∂y(∂z(∂z(ζ))))"
            => ∂x(ζ)^4 + ∂y(ζ)^4 + ∂z(ζ)^4 + 2*∂x(∂x(∂y(∂y(ζ)))) + 2*∂x(∂x(∂z(∂z(ζ)))) + 2*∂y(∂y(∂z(∂z(ζ))))
    )

    for (i, (test_name, op)) in enumerate(test_cases)
        op_computation = Computation(op, dump_field)
        compute!(op_computation)  # precompile

        test_number = @sprintf("%02d", i)
        bname =  benchmark_name(N, "[$test_number] $test_name", arch, nothing)
        @info "Running benchmark: $bname..."
        for _ in 1:Nt
            @timeit timer bname compute!(op_computation)
        end
    end
end

#####
##### Print benchmark results
#####

println()
println(oceananigans_versioninfo())
println(versioninfo_with_gpu())
print_timer(timer, title="Abstract operations benchmarks", sortby=:name)
println()
