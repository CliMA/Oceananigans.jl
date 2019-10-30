#####
##### 'Tupled closure' implementation.
#####
##### For now, we have hacked an implementation for tuples of two closures.
##### We cannot do the case of arbitrary closure tuples until we figure out how to
##### obtain the length of a closure tuple at compile-time on the GPU.
#####

# Stress divergences

for stress_div in (:∂ⱼ_2ν_Σ₁ⱼ, :∂ⱼ_2ν_Σ₂ⱼ, :∂ⱼ_2ν_Σ₃ⱼ)
    @eval begin
        @inline $stress_div(i, j, k, grid, closures::Tuple{C1, C2}, U, Ks) where {C1, C2} = (
              $stress_div(i, j, k, grid, closures[1], U, Ks[1])
            + $stress_div(i, j, k, grid, closures[2], U, Ks[2]))
    end
end

# Tracer flux divergences

@inline ∇_κ_∇c(i, j, k, grid, closures::Tuple{C1, C2}, c, iᶜ, Ks, C, buoyancy) where {C1, C2} = (
      ∇_κ_∇c(i, j, k, grid, closures[1], c, iᶜ, Ks[1], C, buoyancy)
    + ∇_κ_∇c(i, j, k, grid, closures[2], c, iᶜ, Ks[2], C, buoyancy))

# Utilities for closures

function calculate_diffusivities!(Ks, arch, grid, closures::Tuple, args...)
    for (α, closure) in enumerate(closures)
        @inbounds K = Ks[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end
    return nothing
end

TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, tracers, closures::Tuple) =
    Tuple(TurbulentDiffusivities(arch, grid, tracers, closure) for closure in closures)

with_tracers(tracers, closures::Tuple) = Tuple(with_tracers(tracers, closure) for closure in closures)
