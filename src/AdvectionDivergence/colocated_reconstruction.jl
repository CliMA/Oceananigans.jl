abstract type AbstractColocatedReconstructionScheme end
struct TrivialSecondOrder <: AbstractColocatedReconstructionScheme end
struct UpwindWENO4 <: AbstractColocatedReconstructionScheme end
struct CenteredWENO5 <: AbstractColocatedReconstructionScheme end

# Interface for dealing with boundaries (immersed or non-immersed)
@inline _reconstruct_uᶠᵃᵃ(args...) = reconstruct_uᶠᵃᵃ(args...)
@inline _reconstruct_vᵃᶠᵃ(args...) = reconstruct_vᵃᶠᵃ(args...)

@inline reconstruct_uᶠᵃᵃ(i, j, k, grid, ::TrivialSecondOrder, u) = @inbounds u[i, j, k]
@inline reconstruct_vᵃᶠᵃ(i, j, k, grid, ::TrivialSecondOrder, v) = @inbounds v[i, j, k]

