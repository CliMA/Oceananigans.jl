module AbstractOperations

export ∂x, ∂y, ∂z, @at

using Base: @propagate_inbounds

using Oceananigans, Adapt

using Oceananigans: AbstractModel, 
                    AbstractField, AbstractLocatedField, Face, Cell, xnode, ynode, znode,
                    device, launch_config, architecture, 
                    HorizontalAverage, zero_halo_regions!, normalize_horizontal_sum!

import Oceananigans: data, architecture, location, run_diagnostic, AbstractGrid

import Oceananigans.TurbulenceClosures: ∂x_caa, ∂x_faa, ∂y_aca, ∂y_afa, ∂z_aac, ∂z_aaf, 
                                        ▶x_caa, ▶x_faa, ▶y_aca, ▶y_afa, ▶z_aac, ▶z_aaf,
                                        ▶xy_cca, ▶xy_ffa, ▶xy_cfa, ▶xy_fca, 
                                        ▶xz_cac, ▶xz_faf, ▶xz_caf, ▶xz_fac, 
                                        ▶yz_acc, ▶yz_aff, ▶yz_acf, ▶yz_afc,
                                        ▶xyz_ccc, ▶xyz_fcc, ▶xyz_cfc, ▶xyz_ccf,
                                        ▶xyz_fff, ▶xyz_ffc, ▶xyz_fcf, ▶xyz_cff

using GPUifyLoops: @launch, @loop

import Base: getindex

abstract type AbstractOperation{X, Y, Z, G} <: AbstractLocatedField{X, Y, Z, Nothing, G} end

const ALF = AbstractLocatedField

data(op::AbstractOperation) = op
Base.parent(op::AbstractOperation) = op

function validate_grid(a::AbstractField, b::AbstractField)
    a.grid === b.grid || throw(ArgumentError("Two fields in a BinaryOperation must be on the same grid."))
    return a.grid
end

validate_grid(a::AbstractField, b) = a.grid
validate_grid(a, b::AbstractField) = b.grid
validate_grid(a, b) = nothing

function validate_grid(a, b, c...)
    grids = []
    push!(grids, validate_grid(a, b))
    append!(grids, [validate_grid(a, ci) for ci in c])

    for g in grids
        if !(g === nothing)
            return g
        end
    end

    return nothing
end

@inline identity(i, j, k, grid, c) = @inbounds c[i, j, k]
@inline identity(i, j, k, grid, a::Number) = a
@inline identity(i, j, k, grid, F::TF, args...) where TF<:Function = F(i, j, k, grid, args...)

interpolation_code(::Type{Face}) = :f
interpolation_code(::Type{Cell}) = :c
interpolation_code(::Face) = :f
interpolation_code(::Cell) = :c
interpolation_code(from::L, to::L) where L = :a
interpolation_code(from, to) = interpolation_code(to)

for ξ in ("x", "y", "z")
    ▶sym = Symbol(:▶, ξ, :sym)
    @eval begin
        $▶sym(s::Symbol) = $▶sym(Val(s))
        $▶sym(::Union{Val{:f}, Val{:c}}) = $ξ
        $▶sym(::Val{:a}) = ""
    end
end

function interpolation_operator(from, to)
    x, y, z = (interpolation_code(X(), Y()) for (X, Y) in zip(from, to))

    if all(ξ === :a for ξ in (x, y, z))
        return identity
    else 
        return eval(Symbol(:▶, ▶xsym(x), ▶ysym(y), ▶zsym(z), :_, x, y, z))
    end
end

interpolation_operator(::Nothing, to) = identity

# New operators must add to this list
const operators = []

include("unary_operations.jl")
include("binary_operations.jl")
include("polynary_operations.jl")
include("derivatives.jl")
include("computations.jl")
include("function_fields.jl")

function insert_location!(ex::Expr, location)
    if ex.head === :call && ex.args[1] ∈ operators
        push!(ex.args, ex.args[end])
        ex.args[3:end-1] .= ex.args[2:end-2]
        ex.args[2] = location
    end

    for arg in ex.args
        insert_location!(arg, location)
    end

    return nothing
end

insert_location!(anything, location) = nothing

macro at(location, ex)
    insert_location!(ex, location)
    return esc(ex)
end


end # module
