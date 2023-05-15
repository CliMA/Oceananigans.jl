
struct FluxForm{N, FT, A} <: AbstractAdvectionScheme{N, FT}
    advection :: A
    FluxForm{N, FT}(advection::A) where {N, FT, A} = new{N, FT, A}(advection)
end

function FluxForm(FT::DataType=Float64; advection)
    N = boundary_buffer(advection)
    return FluxForm{N, FT}(advection)
end

Adapt.adapt_structure(to, scheme::FluxForm{N, FT}) where {N, FT} =
        FluxForm{N, FT}(Adapt.adapt(to, scheme.advection))

@inline function U_dot_∇u(i, j, k, grid, scheme::FluxForm, U) 

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U.v) / Δxᶠᶜᶜ(i, j, k, grid)
    @inbounds û = U.u[i, j, k]

    return div_𝐯u(i, j, k, grid, scheme.advection, U, U.u) - 
           v̂ * v̂ * δxᶠᶜᶜ(i, j, k, grid, Δyᶜᶜᶜ) / Azᶠᶜᶜ(i, j, k, grid) + 
           v̂ * û * δyᶠᶜᶜ(i, j, k, grid, Δxᶠᶠᶜ) / Azᶠᶜᶜ(i, j, k, grid)
end

@inline function U_dot_∇u(i, j, k, grid, scheme::FluxForm, U) 

    @inbounds û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, u) / Δyᶜᶠᶜ(i, j, k, grid)
    @inbounds v̂ = U.v[i, j, k]

    return div_𝐯u(i, j, k, grid, scheme.advection, U, U.v) + 
           û * v̂ * δxᶜᶠᶜ(i, j, k, grid, Δyᶠᶠᶜ) / Azᶜᶠᶜ(i, j, k, grid) -
           û * û * δyᶜᶠᶜ(i, j, k, grid, Δxᶜᶜᶜ) / Azᶜᶠᶜ(i, j, k, grid)
end
