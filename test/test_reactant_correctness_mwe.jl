using Printf
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations

Reactant.set_default_backend("cpu")

function compare_parent(name, f₁, f₂; rtol=1e-8, atol=sqrt(eps(eltype(f₁))))
    p₁ = Array(parent(f₁))
    p₂ = Array(parent(f₂))

    # debug
    println("\nrank $(rank): vanilla $(p₁)")
    println("rank $(rank): reactant $(p₂)")

    common_sz = map(min, size(p₁), size(p₂))
    v₁ = view(p₁, Base.OneTo.(common_sz)...)
    v₂ = view(p₂, Base.OneTo.(common_sz)...)
    δ = v₁ .- v₂
    max_δ, idx = findmax(abs, δ)
    approx_equal = isapprox(v₁, v₂; rtol, atol)
    @printf("\n(%6s)   parent: ψ₁ ≈ ψ₂: %-5s, max|ψ₁|=%.6e, max|ψ₂|=%.6e, max|δ|=%.6e at %s (overlap %s)\n",
            name, approx_equal, maximum(abs, v₁), maximum(abs, v₂), max_δ, string(idx.I), string(common_sz))
    return approx_equal
end

kw = (size=8, halo=1, extent=1, topology=(Periodic, Flat, Flat))
vanilla_arch = Distributed(CPU())
reactant_arch = Distributed(ReactantState())
vanilla_grid = RectilinearGrid(vanilla_arch; kw...)
reactant_grid = RectilinearGrid(reactant_arch; kw...)

vanilla_field = Field{Center, Center, Center}(vanilla_grid)
reactant_field = Field{Center, Center, Center}(reactant_grid)

# Fill with deterministic, rank-unique sentinel values.
rank = vanilla_arch.local_rank
N = prod(size(vanilla_field))
data = reshape(Float64.(collect(rank * N + 1 : (rank + 1) * N)), size(vanilla_field)...)
set!(vanilla_field, data)
set!(reactant_field, data)

# fill_halo_regions!(vanilla_field)
@jit fill_halo_regions!(reactant_field)

compare_parent("halo", vanilla_field, reactant_field)
