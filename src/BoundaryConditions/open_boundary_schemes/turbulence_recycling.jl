using Oceananigans.Architectures: AbstractSerialArchitecture
using Oceananigans.Grids: nodes, peripheral_node

#####
##### TurbulenceRecycling open boundary scheme
#####

struct TurbulenceRecycling{FT, L, A, D, TF, N, S, R} <: AbstractRadiationScheme{FT}
    recycling_distance :: L # distance between the boundary and the recycling plane (number or nothing)
    averaging :: A
    damping :: D
    target_transport :: TF
    recycling_index :: N # cells between the boundary and the recycling plane (integer or nothing)
    φʳ :: S   # recycling plane times the averaging weight (2D array or nothing)
    Δʳ :: S   # averaging weight on the recycling plane (2D array or nothing)
    φ̄ʳ :: R   # mean of the recycling plane over the averaged dimensions (2D array or nothing)
    ΣΔʳ :: R  # `Δʳ` summed over the averaged dimensions (2D array or nothing)

    function TurbulenceRecycling{FT}(recycling_distance::L, averaging::A, damping::D, target_transport::TF, recycling_index::N,
                                     φʳ::S, Δʳ::S, φ̄ʳ::R, ΣΔʳ::R) where {FT, L, A, D, TF, N, S, R}
        return new{FT, L, A, D, TF, N, S, R}(recycling_distance, averaging, damping, target_transport, recycling_index, φʳ, Δʳ, φ̄ʳ, ΣΔʳ)
    end
end

"""
$(SIGNATURES)

Open boundary scheme that feeds developed turbulence through an inflow boundary by recycling it from the
interior, after [Lund et al. (1998)](@cite lund.ea1998) and [Kataoka and Mizuno (2002)](@cite kataoka.mizuno2002).
Each stage, the plane parallel to the boundary and `recycling_distance`, or `recycling_index` cells, into the
domain is sampled, its mean over the `averaging` dimensions is removed, and the result is added to the exterior
value `φᵉˣᵗ` of the boundary condition:

    φᵇ = φᵉˣᵗ + damping * (φʳ - ⟨φʳ⟩)

The flow between the boundary and the recycling plane is assumed to be statistically homogeneous along the
boundary-normal direction.

`TurbulenceRecycling` is used as the `scheme` of a [`NormalFlowBoundaryCondition`](@ref) for boundary-normal
velocities, where it sets the boundary face, or of a [`ValueBoundaryCondition`](@ref) for fields that are
`Center`-located in the boundary-normal direction, where it sets the halo cell adjacent to the boundary.

Keyword Arguments
=================

- `recycling_distance`: distance between the boundary and the recycling plane, in units of the
  boundary-normal coordinate. The plane is placed on the nearest cell face.
- `recycling_index`: number of cells between the boundary and the recycling plane, given instead of
  `recycling_distance`.
- `averaging`: dimensions over which the mean `⟨φʳ⟩` is taken, as in `Average(field; dims)`, for example `2`
  or `(2, 3)` on a west boundary. The mean is weighted by the grid spacings. With `nothing` no mean is removed.
- `damping`: factor multiplying the recycled signal. A number, an array, or a function with the signature of
  a boundary condition function, for example `damping(y, z, t)` on `x`-boundaries. Default: `1`.
- `target_transport`: net volume transport through the boundary [m³ s⁻¹], positive in the positive coordinate
  direction, to which the boundary-normal velocity is shifted each stage. A number or a callable of the grid.
  Default: `nothing`.

References
==========
* Lund, T. S., Wu, X., & Squires, K. D. (1998). "Generation of turbulent inflow data for
  spatially-developing boundary layer simulations." Journal of Computational Physics, 140(2), 233-258.
* Kataoka, H., & Mizuno, M. (2002). "Numerical flow computation around aeroelastic 3D square cylinder
  using inflow turbulence." Wind and Structures, 5(2-4), 379-392.

```jldoctest
using Oceananigans

TurbulenceRecycling(recycling_distance = 100, averaging = 2)

# output
TurbulenceRecycling{Float64}
├── recycling_distance: 100.0
├── recycling_index: nothing
├── averaging: 2
├── damping: 1
└── target_transport: nothing
```
"""
function TurbulenceRecycling(FT = defaults.FloatType; recycling_distance = nothing, recycling_index = nothing,
                             averaging, damping = 1, target_transport = nothing)

    isnothing(recycling_distance) == isnothing(recycling_index) &&
        throw(ArgumentError("TurbulenceRecycling needs exactly one of recycling_distance and recycling_index."))

    recycling_distance = isnothing(recycling_distance) ? nothing : convert(FT, recycling_distance)
    damping = materialize_condition(damping, nothing, false, ())
    target_transport = convert_target_transport(FT, target_transport)
    return TurbulenceRecycling{FT}(recycling_distance, averaging, damping, target_transport, recycling_index, nothing, nothing, nothing, nothing)
end

Adapt.adapt_structure(to, r::TurbulenceRecycling{FT}) where FT = TurbulenceRecycling{FT}(r.recycling_distance, r.averaging, adapt(to, r.damping),
                                                                                         adapt(to, r.target_transport), r.recycling_index,
                                                                                         adapt(to, r.φʳ), adapt(to, r.Δʳ), adapt(to, r.φ̄ʳ), adapt(to, r.ΣΔʳ))

function Base.show(io::IO, r::TurbulenceRecycling)
    print(io, summary(r), '\n')
    print(io, "├── recycling_distance: ", prettysummary(r.recycling_distance), '\n')
    print(io, "├── recycling_index: ", prettysummary(r.recycling_index), '\n')
    print(io, "├── averaging: ", r.averaging, '\n')
    print(io, "├── damping: ", prettysummary(r.damping), '\n')
    print(io, "└── target_transport: ", prettysummary(r.target_transport))
end

const RecyclingBC = Union{BoundaryCondition{<:Value{<:TurbulenceRecycling}}, BoundaryCondition{<:NormalFlow{<:TurbulenceRecycling}}}

has_target_transport(r::TurbulenceRecycling) = !isnothing(r.target_transport)

function materialize_radiation_storage(r::TurbulenceRecycling{FT}, grid, loc, dim, Side, args...) where FT
    architecture(grid) isa AbstractSerialArchitecture || throw(ArgumentError("TurbulenceRecycling does not support distributed grids."))

    recycling_index = r.recycling_index

    if isnothing(recycling_index)
        faces = on_architecture(CPU(), nodes(grid, Face(), Face(), Face())[dim])
        boundary = Side === LeftBoundary ? first(faces) : last(faces)
        distances = sort(abs.(faces .- boundary))
        recycling_index = argmin(abs.(distances .- r.recycling_distance)) - 1
    end

    0 < recycling_index < size(grid, dim) || throw(ArgumentError("recycling_index = $recycling_index does not place the recycling plane inside the domain."))

    damping = regularize_boundary_condition(r.damping, grid, loc, dim, Side, args...)

    if isnothing(r.averaging)
        buffers = (nothing, nothing, nothing, nothing)
    else
        dim in r.averaging && throw(ArgumentError("averaging = $(r.averaging) includes the boundary-normal dimension $dim."))
        tangential = filter(!=(dim), (1, 2, 3))
        plane = map(d -> size(grid, loc)[d], tangential)
        reduced = map((d, N) -> ifelse(d in r.averaging, 1, N), tangential, plane)
        buffers = (zeros(grid, plane...), zeros(grid, plane...), zeros(grid, reduced...), zeros(grid, reduced...))
    end

    return TurbulenceRecycling{FT}(r.recycling_distance, r.averaging, damping, r.target_transport, recycling_index, buffers...)
end

#####
##### The mean of the recycling plane
#####

# The weight is the product of the spacings along the averaged dimensions, and vanishes on peripheral nodes
@inline function weigh_recycled_state!(side, i, j, grid, c, bc, loc)
    r = bc.classification.scheme
    iʳ, jʳ, kʳ = boundary_point(side, loc, grid, r.recycling_index, i, j)

    Δˣ = Δx(iʳ, jʳ, kʳ, grid, loc...)
    Δʸ = Δy(iʳ, jʳ, kʳ, grid, loc...)
    Δᶻ = Δz(iʳ, jʳ, kʳ, grid, loc...)

    Δ = ifelse(1 in r.averaging, Δˣ, one(Δˣ)) * ifelse(2 in r.averaging, Δʸ, one(Δʸ)) * ifelse(3 in r.averaging, Δᶻ, one(Δᶻ))
    Δ = ifelse(peripheral_node(iʳ, jʳ, kʳ, grid, loc...), zero(Δ), Δ)

    @inbounds r.Δʳ[i, j] = Δ
    @inbounds r.φʳ[i, j] = Δ * c[iʳ, jʳ, kʳ]

    return nothing
end

@inline recycled_mean(::Nothing, i, j, grid) = zero(grid)

# A reduced dimension has length 1
@inline recycled_mean(φ̄ʳ, i, j, grid) = @inbounds φ̄ʳ[min(i, size(φ̄ʳ, 1)), min(j, size(φ̄ʳ, 2))]

#####
##### Stepping the exterior state
#####

@inline function step_exterior_state!(side, i, j, grid, c, bc::RecyclingBC, loc, clock, model_fields, Δt)
    r = bc.classification.scheme
    iᵇ, jᵇ, kᵇ = boundary_point(side, loc, grid, 0, i, j)
    iʳ, jʳ, kʳ = boundary_point(side, loc, grid, r.recycling_index, i, j)

    φᵉˣᵗ = getbc(bc, i, j, grid, clock, model_fields)
    damping = getbc(r.damping, i, j, grid, clock, model_fields)

    φ′ = @inbounds c[iʳ, jʳ, kʳ] - recycled_mean(r.φ̄ʳ, i, j, grid)
    φ′ = ifelse(peripheral_node(iʳ, jʳ, kʳ, grid, loc...), zero(grid), φ′)

    @inbounds c[iᵇ, jᵇ, kᵇ] = φᵉˣᵗ + damping * φ′

    return nothing
end

function step_boundary_condition!(bc::RecyclingBC, side, field, clock, model_fields, Δt)
    r = bc.classification.scheme

    if !isnothing(r.averaging)
        update_exterior_state!(weigh_recycled_state!, bc, side, field)
        sum!(r.φ̄ʳ, r.φʳ)
        sum!(r.ΣΔʳ, r.Δʳ)
        r.φ̄ʳ ./= r.ΣΔʳ
    end

    update_exterior_state!(step_exterior_state!, bc, side, field, clock, model_fields, Δt)

    return nothing
end

cache_boundary_condition!(::RecyclingBC, side, field, clock, model_fields) = nothing

function initialize_boundary_condition!(bc::BoundaryCondition{<:Value{<:TurbulenceRecycling}}, side, field, clock, model_fields)
    return step_boundary_condition!(bc, side, field, clock, model_fields, nothing)
end
