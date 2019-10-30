####
#### 'Tupled closure' implementation
####

# Stress divergences

for stress_div in (:∂ⱼ_2ν_Σ₁ⱼ, :∂ⱼ_2ν_Σ₂ⱼ, :∂ⱼ_2ν_Σ₃ⱼ)
    stress_div_tuple = Symbol(stress_div, :_tuple)
    @eval begin
        @inline function $stress_div(i, j, k, grid, closures::Tuple, U, Ks)
            return $stress_div_tuple(i, j, k, grid, closures, U, Ks)
        end

        @inline function $stress_div_tuple(i, j, k, grid, ct::Tuple{C1, C2}, U, Ks) where {C1, C2}
            return (  $stress_div(i, j, k, grid, closures[1], U, Ks[1])
                    + $stress_div(i, j, k, grid, closures[2], U, Ks[2]))
        end
    end
end

# Tracer flux divergences

@inline function ∇_κ_∇c(i, j, k, grid, closures::Tuple, c, iᶜ, Ks, C, buoyancy) 
    return ∇_κ_∇c_tuple(i, j, k, grid, closures, c, iᶜ, Ks, C, buoyancy)
end
                        
@inline function ∇_κ_∇c_tuple(i, j, k, grid, closures::Tuple{C1, C2}, c, iᶜ, Ks, C, buoyancy) where {C1, C2}
    return (  ∇_κ_∇c(i, j, k, grid, closures[1], c, iᶜ, Ks[1], C, buoyancy)
            + ∇_κ_∇c(i, j, k, grid, closures[2], c, iᶜ, Ks[2], C, buoyancy))
end

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
