#####
##### 'Tupled closure' implementation: 1-tuple, 2-tuple, and then n-tuple by induction
#####

# Stress divergences

for stress_div in (:∂ⱼ_τ₁ⱼ, :∂ⱼ_τ₂ⱼ, :∂ⱼ_τ₃ⱼ)
    @eval begin
        @inline $stress_div(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1}, U, Ks) where {C1} =
                    $stress_div(i, j, k, grid, clock, closures[1], U, Ks[1])

        @inline $stress_div(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1, C2}, U, Ks) where {C1, C2} = (
                    $stress_div(i, j, k, grid, clock, closures[1], U, Ks[1])
                  + $stress_div(i, j, k, grid, clock, closures[2], U, Ks[2]))

        @inline $stress_div(i, j, k, grid::AbstractGrid, clock, closures::Tuple, U, Ks, args...) = (
                    $stress_div(i, j, k, grid, clock, closures[1:2], U, Ks[1:2], args...)
                  + $stress_div(i, j, k, grid, clock, closures[3:end], U, K[3:end], args...))
    end
end

# Tracer flux divergences

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1}, c, iᶜ, Ks, C, buoyancy) where {C1} =
        ∇_dot_qᶜ(i, j, k, grid, clock, closures[1], c, iᶜ, Ks[1], C, buoyancy)

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, clock, closures::Tuple{C1, C2}, c, iᶜ, Ks, C, buoyancy) where {C1, C2} = (
        ∇_dot_qᶜ(i, j, k, grid, clock, closures[1], c, iᶜ, Ks[1], C, buoyancy)
      + ∇_dot_qᶜ(i, j, k, grid, clock, closures[2], c, iᶜ, Ks[2], C, buoyancy))

@inline ∇_dot_qᶜ(i, j, k, grid::AbstractGrid, clock, closures::Tuple, c, iᶜ, Ks, C, buoyancy) = (
        ∇_dot_qᶜ(i, j, k, grid, clock, closures[1:2], c, iᶜ, Ks[1:2], C, buoyancy)
      + ∇_dot_qᶜ(i, j, k, grid, clock, closures[3:end], c, iᶜ, Ks[3:end], C, buoyancy))

# Utilities for closures

function calculate_diffusivities!(Ks, arch, grid, closures::Tuple, args...)
    for (α, closure) in enumerate(closures)
        @inbounds K = Ks[α]
        calculate_diffusivities!(K, arch, grid, closure, args...)
    end
    return nothing
end
