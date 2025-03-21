struct MoistConditionalBuoyancy{FT} <: AbstractBuoyancyFormulation{Nothing}
    background_buoyancy_frequency :: FT
end

required_tracers(::MoistConditionalBuoyancy) = (:D, :M)
Base.summary(b::MoistConditionalBuoyancy{FT}) where FT =
    string("MoistConditionalBuoyancy{$FT}(", prettysummary(b.background_buoyancy_frequency), ")")

function Base.show(io::IO, b::MoistConditionalBuoyancy{FT}) where FT
    N² = b.background_buoyancy_frequency
    print(io, summary(b), '\n',
          "└── background_buoyancy_frequency: ", prettysummary(N²), '\n')
end

const c = Center()
const f = Face()

@inline function buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, b::MoistConditionalBuoyancy, tracers)
    Dᵢ = @inbounds tracers.D[i, j, k] 
    Mᵢ = @inbounds tracers.M[i, j, k]
    N² = b.background_buoyancy_frequency    

    z₁ = znode(i, j, 1, grid, c, c, f)
    zₖ = znode(i, j, k, grid, c, c, f)
    Δz = zₖ - z₁

    return max(Mᵢ, Dᵢ - N² * Δz)
end

