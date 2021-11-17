# Utilities to generate a grid with the following inputs

@inline adapt_if_vector(to, var) = var
@inline adapt_if_vector(to, var::AbstractArray) = Adapt.adapt(to, var)

@inline show_coordinate(Δ::Number, T)            = "Regular, with spacing $Δ"
@inline show_coordinate(Δ::Number, ::Type{Flat}) = "Flattened"
@inline show_coordinate(Δ::AbstractVector, T)    = "Stretched, with spacing min=$(minimum(parent(Δ))), max=$(maximum(parent(Δ)))"

get_domain_extent(coord, N)                 = (coord[1], coord[2])
get_domain_extent(coord::Function, N)       = (coord(1), coord(N+1))
get_domain_extent(coord::AbstractVector, N) = (coord[1], coord[N+1])

get_coord_face(coord::Nothing, i) = 1
get_coord_face(coord::Function, i) = coord(i)
get_coord_face(coord::AbstractVector, i) = CUDA.@allowscalar coord[i]

lower_exterior_Δcoordᶜ(topology,        Fi, Hcoord) = [Fi[end - Hcoord + i] - Fi[end - Hcoord + i - 1] for i = 1:Hcoord]
lower_exterior_Δcoordᶜ(::Type{Bounded}, Fi, Hcoord) = [Fi[2]  - Fi[1] for i = 1:Hcoord]

upper_exterior_Δcoordᶜ(topology,        Fi, Hcoord) = [Fi[i + 1] - Fi[i] for i = 1:Hcoord]
upper_exterior_Δcoordᶜ(::Type{Bounded}, Fi, Hcoord) = [Fi[end]   - Fi[end - 1] for i = 1:Hcoord]

# generate a stretched coordinate passing the explicit coord faces as vector of functionL
function generate_coordinate(FT, topology, N, H, coord, architecture)

    # Ensure correct type for F and derived quantities
    interiorF = zeros(FT, N+1)

    for i = 1:N+1
        interiorF[i] = get_coord_face(coord, i)
    end

    L = interiorF[N+1] - interiorF[1]

    # Build halo regions
    ΔF₋ = lower_exterior_Δcoordᶜ(topology, interiorF, H)
    ΔF₊ = upper_exterior_Δcoordᶜ(topology, interiorF, H)

    c¹, cᴺ⁺¹ = interiorF[1], interiorF[N+1]

    F₋ = [c¹   - sum(ΔF₋[i:H]) for i = 1:H]          # locations of faces in lower halo
    F₊ = reverse([cᴺ⁺¹ + sum(ΔF₊[i:H]) for i = 1:H]) # locations of faces in width of top halo region

    F = vcat(F₋, interiorF, F₊)

    # Build cell centers, cell center spacings, and cell interface spacings
    TC = total_length(Center, topology, N, H)
     C = [ (F[i + 1] + F[i]) / 2 for i = 1:TC ]
    ΔF = [  C[i] - C[i - 1]      for i = 2:TC ]

    # Trim face locations for periodic domains
    TF = total_length(Face, topology, N, H)
    F  = F[1:TF]

    ΔC = [F[i + 1] - F[i] for i = 1:TF-1]

    ΔF = OffsetArray(ΔF, -H)
    ΔC = OffsetArray(ΔC, -H)

    # Seems needed to avoid out-of-bounds error in viscous dissipation
    # operators wanting to access ΔF[N+2].
    ΔF = OffsetArray(cat(ΔF[0], ΔF..., ΔF[N], dims=1), -H-1)

    ΔF = OffsetArray(arch_array(architecture, ΔF.parent), ΔF.offsets...)
    ΔC = OffsetArray(arch_array(architecture, ΔC.parent), ΔC.offsets...)

    F = OffsetArray(F, -H)
    C = OffsetArray(C, -H)

    # Convert to appropriate array type for arch
    F = OffsetArray(arch_array(architecture, F.parent), F.offsets...)
    C = OffsetArray(arch_array(architecture, C.parent), C.offsets...)

    return L, F, C, ΔF, ΔC
end

# generate a regular coordinate passing the domain extent (2-tuple) and number of points
function generate_coordinate(FT, topology, N, H, coord::Tuple{<:Number, <:Number}, architecture)

    @assert length(coord) == 2

    c₁, c₂ = coord
    @assert c₁ < c₂
    L = FT(c₂ - c₁)

    # Convert to get the correct type also when using single precision
    ΔF = ΔC = Δ = convert(FT, L / N)

    F₋ = c₁ - H * Δ
    F₊ = F₋ + total_extent(topology, H, Δ, L)

    C₋ = F₋ + Δ / 2
    C₊ = C₋ + L + Δ * (2H - 1)

    TF = total_length(Face,   topology, N, H)
    TC = total_length(Center, topology, N, H)

    F = range(F₋, F₊, length = TF)
    C = range(C₋, C₊, length = TC)

    F = OffsetArray(F, -H)
    C = OffsetArray(C, -H)
        
    return L, F, C, ΔF, ΔC
end

# Flat domains
function generate_coordinate(FT, ::Type{Flat}, N, H, coord::Tuple{<:Number, <:Number}, architecture)
    return FT(1), range(1, 1, length=N), range(1, 1, length=N), FT(1), FT(1)
end
