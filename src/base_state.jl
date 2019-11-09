struct BaseState{P, R, T}
    p :: P
    ρ :: R
    θ :: T
end

BaseState(; FT=Float64, p=FT(0), ρ=FT(0), θ=FT(0)) = BaseState(p, ρ, θ)

####
#### Pressure base state
####

@inline _base_pressure(i, j, k, grid, p::Number) = p
@inline _base_pressure(i, j, k, grid, p::AbstractArray{T, 1}) where T = @inbounds p[k]
@inline _base_pressure(i, j, k, grid, p::AbstractArray{T, 3}) where T = @inbounds p[i, j, k]
@inline _base_pressure(i, j, k, grid, p::Function) = p(grid.xC[i], grid.yC[j], grid.zC[k])

@inline base_pressure(i, j, k, grid, base_state) = _base_pressure(i, j, k, grid, base_state.p)

####
#### Density base state
####

@inline _base_density(i, j, k, grid, ρ::Number) = ρ
@inline _base_density(i, j, k, grid, ρ::AbstractArray{T, 1}) where T = @inbounds ρ[k]
@inline _base_density(i, j, k, grid, ρ::AbstractArray{T, 3}) where T = @inbounds ρ[i, j, k]
@inline _base_density(i, j, k, grid, ρ::Function) = ρ(grid.xC[i], grid.yC[j], grid.zC[k])

@inline base_density(i, j, k, grid, base_state) = _base_density(i, j, k, grid, base_state.ρ)

####
#### Prognositc temperature base state
####

@inline _base_temperature(i, j, k, grid, θ::Number) = θ
@inline _base_temperature(i, j, k, grid, θ::AbstractArray{T, 1}) where T = @inbounds θ[k]
@inline _base_temperature(i, j, k, grid, θ::AbstractArray{T, 3}) where T = @inbounds θ[i, j, k]
@inline _base_temperature(i, j, k, grid, θ::Function) = θ(grid.xC[i], grid.yC[j], grid.zC[k])

@inline base_temperature(i, j, k, grid, base_state) = _base_density(i, j, k, grid, base_state.ρ) * _base_temperature(i, j, k, grid, base_state.θ)
