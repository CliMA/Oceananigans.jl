using Adapt
using Statistics
using KernelAbstractions
using Oceananigans.Grids
using Oceananigans.BoundaryConditions: zero_halo_regions!

"""
    struct ComputedField{X, Y, Z, A, G, O} <: AbstractField{X, Y, Z, A, G}

Type representing a field computed from an operand.
"""
struct ComputedField{X, Y, Z, A, G, O} <: AbstractField{X, Y, Z, A, G}
       data :: A
       grid :: G
    operand :: O

    function ComputedField{X, Y, Z}(data, grid, operand) where {X, Y, Z}
        validate_field_data(X, Y, Z, data, grid)
        return new{X, Y, Z, typeof(data), typeof(grid), typeof(operand)}(data, grid, operand)
    end
end

"""
    ComputedField(operand; data=nothing)

Returns a field whose data is `computed` from `operand`.
If the keyword argument `data` is not provided, memory is allocated to store
the result. The `arch`itecture of `data` is inferred from `operand`.
"""
function ComputedField(operand; data=nothing)
    
    loc = location(operand)
    arch = architecture(operand)
    grid = operand.grid

    if isnothing(data)
        data = new_data(arch, grid, loc)
    end

    return ComputedField{loc[1], loc[2], loc[3]}(data, grid, operand)
end

"""
    compute!(comp::ComputedField)

Compute `comp.operand` and store the result in `comp.data`.
"""
function compute!(comp::ComputedField{X, Y, Z}) where {X, Y, Z}
    compute!(comp.operand) # ensures any 'dependencies' of the computation are computed first

    arch = architecture(comp.data)

    workgroup, worksize = work_layout(comp.grid,
                                      :xyz,
                                      include_right_boundaries=true,
                                      location=(X, Y, Z))

    compute_kernel! = _compute!(device(arch), workgroup, worksize) 

    event = compute_kernel!(comp.data, comp.operand; dependencies=Event(device(arch)))

    wait(device(arch), event)

    return nothing
end

"""Compute an `operand` and store in `data`."""
@kernel function _compute!(data, operand)
    i, j, k = @index(Global, NTuple)
    @inbounds data[i, j, k] = operand[i, j, k]
end

#####
##### Adapt
#####

Adapt.adapt_structure(to, computed_field::ComputedField{X, Y, Z}) where {X, Y, Z} = 
    ComputedField{X, Y, Z}(Adapt.adapt(to, computed_field.data),
                           Adapt.adapt(to, computed_field.grid),
                           Adapt.adapt(to, computed_field.operand))
