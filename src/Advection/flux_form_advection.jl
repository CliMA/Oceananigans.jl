####
#### Note: FluxForm advection only works for LatitudeLongitude and Rectilinear Grids!
####

struct FluxForm{N, FT, A} <: AbstractAdvectionScheme{N, FT}
    scheme :: A
    FluxForm{N, FT}(scheme::A) where {N, FT, A} = new{N, FT, A}(scheme)
end

function FluxForm(FT::DataType=Float64; scheme)
    N = boundary_buffer(scheme)
    return FluxForm{N, FT}(scheme)
end

Adapt.adapt_structure(to, advection::FluxForm{N, FT}) where {N, FT} =
        FluxForm{N, FT}(Adapt.adapt(to, advection.scheme))

@inline function U_dot_∇u(i, j, k, grid, advection::FluxForm, U) 

    @inbounds v̂ = ℑxᶠᵃᵃ(i, j, k, grid, ℑyᵃᶜᵃ, Δx_qᶜᶠᶜ, U.v) / Δxᶠᶜᶜ(i, j, k, grid)
    @inbounds û = U.u[i, j, k]

    return div_𝐯u(i, j, k, grid, advection.scheme, U, U.u) - 
           v̂ * v̂ * δxᶠᵃᵃ(i, j, k, grid, Δyᶜᶜᶜ) / Azᶠᶜᶜ(i, j, k, grid) + 
           v̂ * û * δyᵃᶜᵃ(i, j, k, grid, Δxᶠᶠᶜ) / Azᶠᶜᶜ(i, j, k, grid)
end

@inline function U_dot_∇v(i, j, k, grid, advection::FluxForm, U) 

    @inbounds û = ℑyᵃᶠᵃ(i, j, k, grid, ℑxᶜᵃᵃ, Δy_qᶠᶜᶜ, U.u) / Δyᶜᶠᶜ(i, j, k, grid)
    @inbounds v̂ = U.v[i, j, k]

    return div_𝐯v(i, j, k, grid, advection.scheme, U, U.v) + 
           û * v̂ * δxᶜᵃᵃ(i, j, k, grid, Δyᶠᶠᶜ) / Azᶜᶠᶜ(i, j, k, grid) -
           û * û * δyᵃᶠᵃ(i, j, k, grid, Δxᶜᶜᶜ) / Azᶜᶠᶜ(i, j, k, grid)
end
