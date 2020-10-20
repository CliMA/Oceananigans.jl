using Adapt
using Statistics
using Oceananigans.Grids
using Oceananigans.Grids: interior_parent_indices

"""
    struct AveragedField{X, Y, Z, A, G, N, O} <: AbstractReducedField{X, Y, Z, A, G, N}

Type representing an average over a field-like object.
"""
struct AveragedField{X, Y, Z, S, A, G, N, O} <: AbstractReducedField{X, Y, Z, A, G, N}
       data :: A
       grid :: G
       dims :: NTuple{N, Int}
    operand :: O
     status :: S

    function AveragedField{X, Y, Z}(data, grid, dims, operand;
                                    recompute_safely=true) where {X, Y, Z}

        dims = validate_reduced_dims(dims)
        validate_reduced_locations(X, Y, Z, dims)
        validate_field_data(X, Y, Z, data, grid)

        status = recompute_safely ? nothing : FieldStatus(0.0)
        
        return new{X, Y, Z, typeof(status), typeof(data),
                   typeof(grid), length(dims), typeof(operand)}(data, grid, dims,
                                                                operand, status)
    end

    function AveragedField{X, Y, Z}(data, grid, dims, operand, status) where {X, Y, Z}
        return new{X, Y, Z, typeof(status), typeof(data),
                   typeof(grid), length(dims), typeof(operand)}(data, grid, dims, operand, status)
    end
end

operand_averaging_indices(avg_loc, op_loc, topo, N, H) = Colon() # fallback when dimension is non-reduced
operand_averaging_indices(::Type{Nothing}, op_loc, topo, N, H) = interior_parent_indices(op_loc, topo, N, H)

"""
    AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=false)

Returns an AveragedField.
"""
function AveragedField(operand::AbstractField; dims, data=nothing, recompute_safely=true)
    
    arch = architecture(operand)
    loc = reduced_location(location(operand), dims=dims)
    grid = operand.grid

    if isnothing(data)
        data = new_data(arch, grid, loc)
        recompute_safely = false
    end

    return AveragedField{loc[1], loc[2], loc[3]}(data, grid, dims, operand,
                                                 recompute_safely=recompute_safely)
end

"""
    compute!(avg::AveragedField)

Compute the average of `avg.operand` and store the result in `avg.data`.
"""
function compute!(avg::AveragedField)
    compute!(avg.operand)

    # Omit halo regions from operand on averaged dimension
    operand_parent = parent(avg.operand)

    Xa, Ya, Za = location(avg)
    Xo, Yo, Zo = location(avg.operand)

    i = operand_averaging_indices(Xa, Xo, topology(avg.grid, 1), avg.grid.Nx, avg.grid.Hx)
    j = operand_averaging_indices(Ya, Yo, topology(avg.grid, 2), avg.grid.Ny, avg.grid.Hy)
    k = operand_averaging_indices(Za, Zo, topology(avg.grid, 3), avg.grid.Nz, avg.grid.Hz)

    operand_sans_halos = view(operand_parent, i, j, k)

    mean!(avg.data.parent, operand_sans_halos)

    return nothing
end

compute!(avg::AveragedField{X, Y, Z, <:FieldStatus}, time) where {X, Y, Z} =
    conditional_compute!(avg, time)

#####
##### Very sugar
#####

Statistics.mean(ϕ::AbstractField; kwargs...) = AveragedField(ϕ; kwargs...)

#####
##### Adapt
#####

Adapt.adapt_structure(to, averaged_field::AveragedField{X, Y, Z}) where {X, Y, Z} = 
    AveragedField{X, Y, Z}(Adapt.adapt(to, averaged_field.data),
                           nothing, averaged_field.dims, nothing, nothing)
