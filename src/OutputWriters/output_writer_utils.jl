 using Oceananigans: AbstractField

#####
#####Output writer utilities
#####

ext(fw::AbstractOutputWriter) = throw("Extension for $(typeof(fw)) is not implemented.")

# When saving stuff to disk like a JLD2 file, `saveproperty!` is used, which
# converts Julia objects to language-agnostic objects.
saveproperty!(file, location, p::Number)        = file[location] = p
saveproperty!(file, location, p::AbstractRange) = file[location] = collect(p)
saveproperty!(file, location, p::AbstractArray) = file[location] = Array(p)
saveproperty!(file, location, p::AbstractField) = file[location] = Array(p.data.parent)
saveproperty!(file, location, p::Function) = @warn "Cannot save Function property into $location"

saveproperty!(file, location, p::Tuple) = [saveproperty!(file, location * "/$i", p[i]) for i in 1:length(p)]

saveproperty!(file, location, p) = [saveproperty!(file, location * "/$subp", getproperty(p, subp))
                                        for subp in propertynames(p)]

# Special saveproperty! so boundary conditions are easily readable outside julia.
function saveproperty!(file, location, cbcs::CoordinateBoundaryConditions)
    for endpoint in propertynames(cbcs)
        endpoint_bc = getproperty(cbcs, endpoint)
        if isa(endpoint_bc.condition, Function)
            @warn "$field.$coord.$endpoint boundary is of type Function and cannot be saved to disk!"
            file["boundary_conditions/$field/$coord/$endpoint/type"] = string(bctype(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = missing
        else
            file["boundary_conditions/$field/$coord/$endpoint/type"] = string(bctype(endpoint_bc))
            file["boundary_conditions/$field/$coord/$endpoint/condition"] = endpoint_bc.condition
        end
    end
end

saveproperties!(file, structure, ps) = [saveproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# When checkpointing, `serializeproperty!` is used, which serializes objects
# unless they need to be converted (basically CuArrays only).
serializeproperty!(file, location, p) = file[location] = p
serializeproperty!(file, location, p::Union{AbstractArray, AbstractField}) = saveproperty!(file, location, p)
serializeproperty!(file, location, p::Function) = @warn "Cannot serialize Function property into $location"

serializeproperties!(file, structure, ps) = [serializeproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# Don't check arrays because we don't need that noise.
has_reference(T, ::AbstractArray{<:Number}) = false

# These two conditions are true, but should not necessary.
has_reference(::Type{Function}, ::AbstractField) = false
has_reference(::Type{T}, ::NTuple{N, <:T}) where {N, T} = true

"""
    has_reference(has_type, obj)

Check (or attempt to check) if `obj` contains, somewhere among its
subfields and subfields of fields, a reference to an object of type
`has_type`. This function doesn't always work.
"""
function has_reference(has_type, obj)
    if typeof(obj) <: has_type
        return true
    elseif applicable(iterate, obj) && length(obj) > 1
        return any([has_reference(has_type, elem) for elem in obj])
    elseif applicable(propertynames, obj) && length(propertynames(obj)) > 0
        return any([has_reference(has_type, getproperty(obj, p)) for p in propertynames(obj)])
    else
        return typeof(obj) <: has_type
    end
end
