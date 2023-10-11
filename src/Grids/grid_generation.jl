# Utilities to generate a grid with the following inputs

@inline adapt_if_vector(to, var) = var
@inline adapt_if_vector(to, var::AbstractArray) = Adapt.adapt(to, var)

get_domain_extent(::Nothing, N)             = (1, 1)
get_domain_extent(coord, N)                 = (coord[1], coord[2])
get_domain_extent(coord::Function, N)       = (coord(1), coord(N+1))
get_domain_extent(coord::AbstractVector, N) = CUDA.@allowscalar (coord[1], coord[N+1])

get_face_node(coord::Nothing, i) = 1
get_face_node(coord::Function, i) = coord(i)
get_face_node(coord::AbstractVector, i) = CUDA.@allowscalar coord[i]

const AT = AbstractTopology
lower_exterior_Δcoordᶠ(::AT,              Fi, Hcoord) = [Fi[end - Hcoord + i] - Fi[end - Hcoord + i - 1] for i = 1:Hcoord]
lower_exterior_Δcoordᶠ(::BoundedTopology, Fi, Hcoord) = [Fi[2]  - Fi[1] for _ = 1:Hcoord]

upper_exterior_Δcoordᶠ(::AT,              Fi, Hcoord) = [Fi[i + 1] - Fi[i] for i = 1:Hcoord]
upper_exterior_Δcoordᶠ(::BoundedTopology, Fi, Hcoord) = [Fi[end]   - Fi[end - 1] for _ = 1:Hcoord]

upper_interior_F(::AT, coord, Δ)           = coord - Δ
upper_interior_F(::BoundedTopology, coord) = coord

total_interior_length(::AT, N)              = N
total_interior_length(::BoundedTopology, N) = N + 1

bad_coordinate_message(ξ::Function, name) = "The values of $name(index) must increase as the index increases!"
bad_coordinate_message(ξ::AbstractArray, name) = "The elements of $name must be increasing!"

# generate a variably-spaced coordinate passing the explicit coord faces as vector or function
function generate_coordinate(FT, topo::AT, N, H, node_generator, coordinate_name, arch)

    # Ensure correct type for F and derived quantities
    interior_face_nodes = zeros(FT, N+1)

    # Use the user-supplied "generator" to build the interior nodes
    for idx = 1:N+1
        interior_face_nodes[idx] = get_face_node(node_generator, idx)
    end

    # Check that the interior nodes are increasing
    if !issorted(interior_face_nodes)
        msg = bad_coordinate_message(node_generator, coordinate_name)
        throw(ArgumentError(msg))
    end

    # Get domain extent
    L = interior_face_nodes[N+1] - interior_face_nodes[1]

    # Build halo regions: spacings first
    Δᶠ₋ = lower_exterior_Δcoordᶠ(topo, interior_face_nodes, H)
    Δᶠ₊ = reverse(upper_exterior_Δcoordᶠ(topo, interior_face_nodes, H))

    c¹, cᴺ⁺¹ = interior_face_nodes[1], interior_face_nodes[N+1]

    F₋ =         [c¹   - sum(Δᶠ₋[i:H]) for i = 1:H]  # locations of faces in lower halo
    F₊ = reverse([cᴺ⁺¹ + sum(Δᶠ₊[i:H]) for i = 1:H]) # locations of faces in top halo

    F = vcat(F₋, interior_face_nodes, F₊)

    # Build cell centers, cell center spacings, and cell interface spacings
    TC = total_length(Center(), topo, N, H)
     C = [(F[i + 1] + F[i]) / 2 for i = 1:TC]
    Δᶠ = [ C[i] - C[i - 1]      for i = 2:TC]

    # Trim face locations for periodic domains
    TF = total_length(Face(), topo, N, H)
    F  = F[1:TF]

    Δᶜ = [F[i + 1] - F[i] for i = 1:TF-1]

    Δᶠ = [Δᶠ[1], Δᶠ..., Δᶠ[end]]
    for i = length(Δᶠ):-1:2
        Δᶠ[i] = Δᶠ[i-1]
    end

    Δᶜ = OffsetArray(arch_array(arch, Δᶜ), -H)
    Δᶠ = OffsetArray(arch_array(arch, Δᶠ), -H-1)

    F = OffsetArray(F, -H)
    C = OffsetArray(C, -H)

    # Convert to appropriate array type for arch
    F = OffsetArray(arch_array(arch, F.parent), F.offsets...)
    C = OffsetArray(arch_array(arch, C.parent), C.offsets...)

    return L, F, C, Δᶠ, Δᶜ
end

# Generate a regularly-spaced coordinate passing the domain extent (2-tuple) and number of points
function generate_coordinate(FT, topo::AT, N, H, node_interval::Tuple{<:Number, <:Number}, coordinate_name, arch)

    if node_interval[2] < node_interval[1]
        msg = "$coordinate_name must be an increasing interval!"
        throw(ArgumentError(msg))
    end

    c₁, c₂ = @. BigFloat(node_interval)
    @assert c₁ < c₂
    L = c₂ - c₁

    # Convert to get the correct type also when using single precision
    Δᶠ = Δᶜ = Δ = L / N

    F₋ = c₁ - H * Δ
    F₊ = F₋ + total_extent(topo, H, Δ, L)

    C₋ = F₋ + Δ / 2
    C₊ = C₋ + L + Δ * (2H - 1)

    TF = total_length(Face(),   topo, N, H)
    TC = total_length(Center(), topo, N, H)

    F = range(FT(F₋), FT(F₊), length = TF)
    C = range(FT(C₋), FT(C₊), length = TC)

    F = OffsetArray(F, -H)
    C = OffsetArray(C, -H)

    return FT(L), F, C, FT(Δᶠ), FT(Δᶜ)
end

# Flat domains
generate_coordinate(FT, ::Flat, N, H, coord::Tuple{<:Number, <:Number}, arch) =
    FT(1), range(1, 1, length=N), range(1, 1, length=N), FT(1), FT(1)
