"""
    ϵ(r, β, L)


"""
@inline ϵ(r, β, L) = r / (β*L)

"""
    ψ_Stommel(x, y; τ₀, β, r)

Stommel gyre streamfunction taken from equation (19.39) of Vallis (2nd edition) where
    * L is the length of each side of the square domain
    * τ₀ is the magnitude of the wind stress
    * β is the Rossby parameter
    * r is the linear drag, or Rayleigh friction, acting on the vertically integrated velocity.
"""
@inline ψ_Stommel(x, y; L, τ₀, β, r) = τ₀*π/β * (1 - x/L - exp(-x/L/ϵ(r, β, L))) * sin(π*y/L)

@inline u_Stommel(x, y; L, τ₀, β, r) = τ₀*π^2/β/L * (1 - x/L - exp(-x/L/ϵ(r, β, L))) * cos(π*y/L)

@inline v_Stommel(x, y; L, τ₀, β, r) = τ₀*π/β/L * (1 - 1/ϵ(r, β, L) * exp(-x/L/ϵ(r, β, L))) * sin(π*y/L)
