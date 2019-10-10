module AbstractOperations

export ∂x, ∂y, ∂z

using Base: @propagate_inbounds

using Oceananigans

using Oceananigans: Face, Cell, AbstractLocatedField, device, launch_config, architecture

using GPUifyLoops: @launch, @loop

import Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                        ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf

import Base: *, -, +, /, getindex

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

data(op::AbstractOperation) = op
alldata(op::AbstractOperation) = op
alldata(a::Field) = a.data

@inline ∂x_caa(i, j, k, grid, u::AbstractOperation) = @inbounds (u[i+1, j, k] - u[i, j, k]) / grid.Δx
@inline ∂x_faa(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k] - c[i-1, j, k]) / grid.Δx

@inline ∂y_aca(i, j, k, grid, v::AbstractOperation) = @inbounds (v[i, j+1, k] - v[i, j, k]) / grid.Δy
@inline ∂y_afa(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k] - c[i, j-1, k]) / grid.Δy

@inline ∂z_aac(i, j, k, grid, w::AbstractOperation) = @inbounds (w[i, j, k] - w[i, j, k+1]) / grid.Δz
@inline ∂z_aaf(i, j, k, grid, c::AbstractOperation) = @inbounds (c[i, j, k-1] - c[i, j, k]) / grid.Δz

@inline ▶x_faa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i-1, j, k])

@inline ▶x_caa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i+1, j, k])

@inline ▶y_afa(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j-1, k])

@inline ▶y_aca(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j+1, k])

@inline ▶z_aaf(i, j, k, grid::RegularCartesianGrid{FT}, F::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (F[i, j, k] + F[i, j, k-1])

@inline ▶z_aac(i, j, k, grid::RegularCartesianGrid{FT}, w::AbstractOperation, args...) where FT =
    @inbounds FT(0.5) * (w[i, j, k] + w[i, j, k+1])

include("binary_operation.jl")
include("derivative.jl")

end # module
