#####
##### Seam topology interface
#####
##### A seam transform has two independent parts:
#####   1. an index permutation that maps destination halo indices to source indices,
#####   2. a signed vector rotation that maps destination vector components to source
#####      vector components.
#####
##### This interface is independent of a particular spherical topology. It is meant
##### to describe folded tripolar / zipper boundaries, cubed-sphere panel seams,
##### and OctaHEALPix folds with the same index-and-vector transform contract.
#####

abstract type AbstractSeamIndexPermutation end
abstract type AbstractSeamVectorRotation end
abstract type AbstractSeamTransform end
abstract type AbstractVectorSeamTransform <: AbstractSeamTransform end

struct SeamIndexPermutation{Swap, ReverseFirst, ReverseSecond} <: AbstractSeamIndexPermutation end
struct SeamVectorRotation{A₁₁, A₁₂, A₂₁, A₂₂} <: AbstractSeamVectorRotation end

struct ScalarSeamTransform{P} <: AbstractSeamTransform
    index_permutation :: P
end

struct CovariantVectorSeamTransform{P, R} <: AbstractVectorSeamTransform
    index_permutation :: P
    vector_rotation :: R
end

struct ContravariantVectorSeamTransform{P, R} <: AbstractVectorSeamTransform
    index_permutation :: P
    vector_rotation :: R
end

const IdentitySeamIndexPermutation = SeamIndexPermutation{false, false, false}
const IdentitySeamVectorRotation = SeamVectorRotation{1, 0, 0, 1}

@inline SeamIndexPermutation(::Val{Swap},
                             ::Val{ReverseFirst},
                             ::Val{ReverseSecond}) where {Swap, ReverseFirst, ReverseSecond} =
    SeamIndexPermutation{Swap, ReverseFirst, ReverseSecond}()

@inline seam_index_permutation(::Val{:identity}) =
    IdentitySeamIndexPermutation()

@inline seam_index_permutation(::Val{:reverse_first}) =
    SeamIndexPermutation{false, true, false}()

@inline seam_index_permutation(::Val{:reverse_second}) =
    SeamIndexPermutation{false, false, true}()

@inline seam_index_permutation(::Val{:reverse_both}) =
    SeamIndexPermutation{false, true, true}()

@inline seam_index_permutation(::Val{:swap}) =
    SeamIndexPermutation{true, false, false}()

@inline seam_index_permutation(::Val{:swap_reverse_first}) =
    SeamIndexPermutation{true, true, false}()

@inline seam_index_permutation(::Val{:swap_reverse_second}) =
    SeamIndexPermutation{true, false, true}()

@inline seam_index_permutation(::Val{:swap_reverse_both}) =
    SeamIndexPermutation{true, true, true}()

@inline ScalarSeamTransform() =
    ScalarSeamTransform(IdentitySeamIndexPermutation())

@inline ScalarSeamTransform(index_permutation::Val) =
    ScalarSeamTransform(seam_index_permutation(index_permutation))

@inline CovariantVectorSeamTransform() =
    CovariantVectorSeamTransform(IdentitySeamIndexPermutation(), IdentitySeamVectorRotation())

@inline ContravariantVectorSeamTransform() =
    ContravariantVectorSeamTransform(IdentitySeamIndexPermutation(), IdentitySeamVectorRotation())

@inline CovariantVectorSeamTransform(index_permutation::Val, rotation::Val) =
    CovariantVectorSeamTransform(seam_index_permutation(index_permutation), seam_vector_rotation(rotation))

@inline ContravariantVectorSeamTransform(index_permutation::Val, rotation::Val) =
    ContravariantVectorSeamTransform(seam_index_permutation(index_permutation), seam_vector_rotation(rotation))

@inline CovariantVectorSeamTransform(index_permutation, rotation::Val) =
    CovariantVectorSeamTransform(index_permutation, seam_vector_rotation(rotation))

@inline ContravariantVectorSeamTransform(index_permutation, rotation::Val) =
    ContravariantVectorSeamTransform(index_permutation, seam_vector_rotation(rotation))

@inline vector_seam_transform(::Val{:covariant}) =
    CovariantVectorSeamTransform()

@inline vector_seam_transform(::Val{:contravariant}) =
    ContravariantVectorSeamTransform()

@inline vector_seam_transform(transform::AbstractVectorSeamTransform) =
    transform

@inline scalar_seam_transform(transform::AbstractSeamTransform) =
    ScalarSeamTransform(seam_index_permutation(transform))

function seam_source_indices end
function seam_vector_source_indices_and_sign end

@inline seam_index_permutation(transform::AbstractSeamTransform) =
    transform.index_permutation

@inline seam_vector_rotation(transform::AbstractVectorSeamTransform) =
    transform.vector_rotation

@inline seam_vector_rotation(::Val{0}) = SeamVectorRotation{ 1,  0,  0,  1}()
@inline seam_vector_rotation(::Val{1}) = SeamVectorRotation{ 0, -1,  1,  0}()
@inline seam_vector_rotation(::Val{2}) = SeamVectorRotation{-1,  0,  0, -1}()
@inline seam_vector_rotation(::Val{3}) = SeamVectorRotation{ 0,  1, -1,  0}()

@inline seam_vector_rotation(::Val{:identity}) = seam_vector_rotation(Val(0))
@inline seam_vector_rotation(::Val{:rotate_90}) = seam_vector_rotation(Val(1))
@inline seam_vector_rotation(::Val{:rotate_180}) = seam_vector_rotation(Val(2))
@inline seam_vector_rotation(::Val{:rotate_270}) = seam_vector_rotation(Val(3))

@inline seam_vector_rotation(::Val{:reflect_first}) = SeamVectorRotation{-1,  0,  0,  1}()
@inline seam_vector_rotation(::Val{:reflect_second}) = SeamVectorRotation{ 1,  0,  0, -1}()
@inline seam_vector_rotation(::Val{:reflect_diagonal}) = SeamVectorRotation{ 0,  1,  1,  0}()
@inline seam_vector_rotation(::Val{:reflect_antidiagonal}) = SeamVectorRotation{ 0, -1, -1,  0}()

@inline function seam_permute_indices(i, j, Nx, Ny,
                                      ::SeamIndexPermutation{Swap, ReverseFirst, ReverseSecond}) where {Swap, ReverseFirst, ReverseSecond}
    first_index = ifelse(Swap, j, i)
    second_index = ifelse(Swap, i, j)
    first_size = ifelse(Swap, Ny, Nx)
    second_size = ifelse(Swap, Nx, Ny)

    source_i = ifelse(ReverseFirst, first_size + 1 - first_index, first_index)
    source_j = ifelse(ReverseSecond, second_size + 1 - second_index, second_index)

    return source_i, source_j
end

@inline seam_transform_indices(i, j, Nx, Ny, transform::AbstractSeamTransform) =
    seam_permute_indices(i, j, Nx, Ny, seam_index_permutation(transform))

@inline function seam_vector_source_indices_and_sign(i, j, Nx, Ny, topology)
    source_i, source_j = seam_source_indices(i, j, Nx, Ny, topology)
    return source_i, source_j, 1
end

@inline function seam_halo_source(i, j, Nx, Ny, topology, transform::ScalarSeamTransform)
    source_i, source_j = seam_source_indices(i, j, Nx, Ny, topology)
    return seam_transform_indices(source_i, source_j, Nx, Ny, transform)
end

@inline vector_source_component_and_sign(::Val{:u},
                                         ::SeamVectorRotation{A₁₁, A₁₂, A₂₁, A₂₂}) where {A₁₁, A₁₂, A₂₁, A₂₂} =
    ifelse(A₁₁ == 0, 2, 1), ifelse(A₁₁ == 0, A₁₂, A₁₁)

@inline vector_source_component_and_sign(::Val{:v},
                                         ::SeamVectorRotation{A₁₁, A₁₂, A₂₁, A₂₂}) where {A₁₁, A₁₂, A₂₁, A₂₂} =
    ifelse(A₂₁ == 0, 2, 1), ifelse(A₂₁ == 0, A₂₂, A₂₁)

@inline seam_vector_source_component_and_sign(component, transform::AbstractVectorSeamTransform) =
    vector_source_component_and_sign(component, seam_vector_rotation(transform))

@inline function seam_halo_source(i, j, Nx, Ny, topology,
                                  component,
                                  transform::AbstractVectorSeamTransform)
    source_i, source_j, topology_sign = seam_vector_source_indices_and_sign(i, j, Nx, Ny, topology)
    source_i, source_j = seam_transform_indices(source_i, source_j, Nx, Ny, transform)
    source_component, sign = seam_vector_source_component_and_sign(component, transform)
    return source_component, source_i, source_j, topology_sign * sign
end
