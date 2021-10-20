# Utilities to generate a grid with the following inputs

@inline adapt_if_vector(var::Nothing)        = nothing
@inline adapt_if_vector(var)                 = var
@inline adapt_if_vector(var::AbstractVector) = Adapt.adapt(to, var)

function show_coordinate(Δ::FC) where {FC<:Number} 
    string = "Regular, with spacing $Δ"
    return string
end  

function show_coordinate(Δ::FC) where {FC<:AbstractVector} 
    Δₘᵢₙ = minimum(parent(Δ))
    Δₘₐₓ = maximum(parent(Δ))
    string = "Stretched, with spacing min=$Δₘᵢₙ, max=$Δₘₐₓ"
    return string
end  

get_domain_extent(coord, N)                 = (coord[1], coord[2])
get_domain_extent(coord::Function, N)       = (coord(1), coord(N+1))
get_domain_extent(coord::AbstractVector, N) = (coord[1], coord[N+1])

get_coord_face(coord::Function, k) = coord(k)
get_coord_face(coord::AbstractVector, k) = CUDA.@allowscalar coord[k]

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

    F₋ = [c¹   - sum(ΔF₋[i:H]) for i = 1:H] # locations of faces in lower halo
    F₊ = reverse([cᴺ⁺¹ + sum(ΔF₊[i:H]) for i = 1:H]) # locations of faces in width of top halo region

    F = vcat(F₋, interiorF, F₊)

    # Build cell centers, cell center spacings, and cell interface spacings
    TC = total_length(Center, topology, N, H)
     C = [ (F[i + 1] + F[i]) / 2 for i = 1:TC ]
    ΔC = [  C[i] - C[i - 1]      for i = 2:TC ]

    # Trim face locations for periodic domains
    TF = total_length(Face, topology, N, H)
    F  = F[1:TF]

    ΔF = [F[i + 1] - F[i] for i = 1:TF-1]

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
    L = c₂ - c₁

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

@inline hack_cosd(φ) = cos(π * φ / 180)
@inline hack_sind(φ) = sin(π * φ / 180)

function generate_curvilinear_operators(FT, Δλᶠ, Δλᶜ, Δφᶠ, φᶠ, φᶜ, radius)
          
    # preallocate quantitie to ensure correct type and size
    Δyᶜᶠᵃ = OffsetArray(zeros(FT, length(Δφᶠ)), φᶠ.offsets[1])
    
    Δxᶜᶠᵃ = OffsetArray(zeros(FT, length(Δλᶜ), length(φᶠ)), Δλᶜ.offsets[1], φᶠ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(zeros(FT, length(Δλᶠ), length(φᶜ)), Δλᶠ.offsets[1], φᶜ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(zeros(FT, length(Δλᶜ), length(φᶠ)), Δλᶜ.offsets[1], φᶠ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(zeros(FT, length(Δλᶠ), length(φᶜ)), Δλᶠ.offsets[1], φᶜ.offsets[1])
       
    for (i,x) in pairs(Δλᶠ)
        for (j,x) in pairs(φᶜ)
            Azᶠᶠᵃ[i, j] = @inbounds radius^2 * deg2rad(Δλᶠ[i]) * (hack_sind(φᶜ[j])   - hack_sind(φᶜ[j-1]))
            Δxᶠᶜᵃ[i, j] = @inbounds radius * hack_cosd(φᶜ[j]) * deg2rad(Δλᶠ[i])
        end 
    end 
    
    for (i,x) in pairs(Δλᶜ)
        for (j,x) in pairs(φᶠ)
            Δxᶜᶠᵃ[i, j] = @inbounds radius * hack_cosd(φᶠ[j]) * deg2rad(Δλᶜ[i])
            Azᶜᶜᵃ[i, j] = @inbounds radius^2 * deg2rad(Δλᶜ[i]) * (hack_sind(φᶠ[j+1]) - hack_sind(φᶠ[j]))
        end
    end

    for (j,x) in pairs(Δφᶠ)
        Δyᶜᶠᵃ = convert(FT, radius * deg2rad(Δφᶠ[j]))
    end

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function generate_curvilinear_operators(FT, Δλᶠ::Number, Δλᶜ, Δφᶠ, φᶠ, φᶜ, radius)
          
    # preallocate quantitie to ensure correct type and size
    Δyᶜᶠᵃ = OffsetArray(zeros(FT, length(Δφᶠ)), φᶠ.offsets[1])
    Δxᶜᶠᵃ = OffsetArray(zeros(FT, length(φᶠ)),  φᶠ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(zeros(FT, length(φᶜ)),  φᶜ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(zeros(FT, length(φᶠ)),  φᶠ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(zeros(FT, length(φᶜ)),  φᶜ.offsets[1])
    
    for (j,x) in pairs(φᶠ)
        Δxᶜᶠᵃ[j] = @inbounds radius * hack_cosd(φᶠ[j]) * deg2rad(Δλᶜ)
        Azᶜᶜᵃ[j] = @inbounds radius^2 * deg2rad(Δλᶜ) * (hack_sind(φᶠ[j+1]) - hack_sind(φᶠ[j]))
    end
    
    for (j,x) in pairs(φᶜ)
        Azᶠᶠᵃ[j] = @inbounds radius^2 * deg2rad(Δλᶠ) * (hack_sind(φᶜ[j])   - hack_sind(φᶜ[j-1]))
        Δxᶠᶜᵃ[j] = @inbounds radius * hack_cosd(φᶜ[j]) * deg2rad(Δλᶠ)
    end 
    for (j,x) in pairs(φᶠ)
        Δyᶜᶠᵃ = convert(FT, radius * deg2rad(Δφᶠ[j]))
    end

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function generate_curvilinear_operators(FT, Δλᶠ, Δλᶜ, Δφᶠ::Number, φᶠ, φᶜ, radius)
    
    Δyᶜᶠᵃ = convert(FT, radius * deg2rad(Δφᶠ))
    
    # preallocate quantitie to ensure correct type and size
    Δxᶜᶠᵃ = OffsetArray(zeros(FT, length(Δλᶜ), length(φᶠ)), Δλᶜ.offsets[1], φᶠ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(zeros(FT, length(Δλᶠ), length(φᶜ)), Δλᶠ.offsets[1], φᶜ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(zeros(FT, length(Δλᶜ), length(φᶠ)), Δλᶜ.offsets[1], φᶠ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(zeros(FT, length(Δλᶠ), length(φᶜ)), Δλᶠ.offsets[1], φᶜ.offsets[1])
    
    for (i,x) in pairs(Δλᶠ)
        for (j,x) in pairs(φᶜ)
            Azᶠᶠᵃ[i, j] = @inbounds radius^2 * deg2rad(Δλᶠ[i]) * (hack_sind(φᶜ[j])   - hack_sind(φᶜ[j-1]))
            Δxᶠᶜᵃ[i, j] = @inbounds radius * hack_cosd(φᶜ[j]) * deg2rad(Δλᶠ[i])
        end 
    end 
    
    for (i,x) in pairs(Δλᶜ)
        for (j,x) in pairs(φᶠ)
            Δxᶜᶠᵃ[i, j] = @inbounds radius * hack_cosd(φᶠ[j]) * deg2rad(Δλᶜ[i])
            Azᶜᶜᵃ[i, j] = @inbounds radius^2 * deg2rad(Δλᶜ[i]) * (hack_sind(φᶠ[j+1]) - hack_sind(φᶠ[j]))
        end
    end

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end

function generate_curvilinear_operators(FT, Δλᶠ::Number, Δλᶜ, Δφᶠ::Number, φᶠ, φᶜ, radius)
    
    Δyᶜᶠᵃ = convert(FT, @inbounds radius * deg2rad(Δφᶠ))
    
    # preallocate quantitie to ensure correct type and size
    Δxᶜᶠᵃ = OffsetArray(zeros(FT, length(φᶠ)), φᶠ.offsets[1])
    Δxᶠᶜᵃ = OffsetArray(zeros(FT, length(φᶜ)), φᶜ.offsets[1])
    Azᶜᶜᵃ = OffsetArray(zeros(FT, length(φᶠ)), φᶠ.offsets[1])
    Azᶠᶠᵃ = OffsetArray(zeros(FT, length(φᶜ)), φᶜ.offsets[1])
    
    for (j,x) in pairs(φᶠ)
        Δxᶜᶠᵃ[j] = @inbounds radius * hack_cosd(φᶠ[j]) * deg2rad(Δλᶜ)
        Azᶜᶜᵃ[j] = @inbounds radius^2 * deg2rad(Δλᶜ) * (hack_sind(φᶠ[j+1]) - hack_sind(φᶠ[j]))
    end
    
    for (j,x) in pairs(φᶜ)
        Azᶠᶠᵃ[j] = @inbounds radius^2 * deg2rad(Δλᶠ) * (hack_sind(φᶜ[j])   - hack_sind(φᶜ[j-1]))
        Δxᶠᶜᵃ[j] = @inbounds radius * hack_cosd(φᶜ[j]) * deg2rad(Δλᶠ)
    end 

    return Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ
end


