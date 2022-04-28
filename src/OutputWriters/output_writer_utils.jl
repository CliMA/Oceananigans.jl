using StructArrays: StructArray, replace_storage
using Oceananigans.Grids: on_architecture
using Oceananigans.Fields: AbstractField, indices, boundary_conditions, instantiated_location
using Oceananigans.BoundaryConditions: bc_str, FieldBoundaryConditions, ContinuousBoundaryFunction, DiscreteBoundaryFunction
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper
using Oceananigans.LagrangianParticleTracking: LagrangianParticles

#####
##### Output writer utilities
#####

"""
    ext(ow)

Return the file extension for the output writer or output
writer type `ow`.
"""
ext(ow::Type{AbstractOutputWriter}) = throw("Extension for $ow is not implemented.")
ext(ow::AbstractOutputWriter) = ext(typeof(fw))

# TODO: add example to docstring below

"""
    saveproperty!(file, address, obj)

Save data in `obj` to `file[address]` in a "languate-agnostic" way,
thus primarily consisting of arrays and numbers, absent Julia-specific types
or other data that can _only_ be interpreted by Julia.
"""
saveproperty!(file, address, obj) = _saveproperty!(file, address, obj)

# Generic implementation: recursively unwrap an object.
_saveproperty!(file, address, obj) = [saveproperty!(file, address * "/$prop", getproperty(obj, prop)) for prop in propertynames(obj)]

# Some specific things
saveproperty!(file, address, p::Union{Number, Array}) = file[address] = p
saveproperty!(file, address, p::AbstractRange)        = file[address] = collect(p)
saveproperty!(file, address, p::AbstractArray)        = file[address] = Array(parent(p))
saveproperty!(file, address, p::Function)             = nothing
saveproperty!(file, address, p::Tuple)                = [saveproperty!(file, address * "/$i", p[i]) for i in 1:length(p)]
saveproperty!(file, address, grid::AbstractGrid)      = _saveproperty!(file, address, on_architecture(CPU(), grid))

# Special saveproperty! so boundary conditions are easily readable outside julia.
function saveproperty!(file, address, bcs::FieldBoundaryConditions)
    for boundary in propertynames(bcs)
        bc = getproperty(bcs, endpoint)
        file[address * "/$endpoint/type"] = bc_str(bc)

        if bc.condition isa Function || bc.condition isa ContinuousBoundaryFunction
            file[address * "/$boundary/condition"] = missing
        else
            file[address * "/$boundary/condition"] = bc.condition
        end
    end
end

"""
    serializeproperty!(file, address, obj)

Serialize `obj` to `file[address]` in a "friendly" way; i.e. converting
`CuArray` to `Array` so data can be loaded on any architecture,
and not attempting to serialize objects that generally aren't
deserializable, like `Function`.
"""
serializeproperty!(file, address, p)                = file[address] = p
serializeproperty!(file, address, p::AbstractArray) = saveproperty!(file, address, p)

const CantSerializeThis = Union{Function,
                                ContinuousBoundaryFunction,
                                DiscreteBoundaryFunction}

serializeproperty!(file, address, p::CantSerializeThis) = nothing

# Convert to CPU please!
# TODO: use on_architecture for more stuff?
serializeproperty!(file, address, grid::AbstractGrid) = file[address] = on_architecture(CPU(), grid)

function serializeproperty!(file, address, p::FieldBoundaryConditions)
    # TODO: it'd be better to "filter" `FieldBoundaryCondition` and then serialize
    # rather than punting with `missing` instead.
    if has_reference(Function, p)
        file[address] = missing
    else
        file[address] = p
    end
end

function serializeproperty!(file, address, f::Field)
    serializeproperty!(file, address * "/location", instantiated_location(f))
    serializeproperty!(file, address * "/data", parent(f))
    serializeproperty!(file, address * "/indices", indices(f))
    serializeproperty!(file, address * "/boundary_conditions", boundary_conditions(f))
    return nothing
end

# Special serializeproperty! for AB2 time stepper struct used by the checkpointer so
# it only saves the fields and not the tendency BCs or χ value (as they can be
# constructed by the `Model` constructor).
function serializeproperty!(file, address, ts::RungeKutta3TimeStepper)
    serializeproperty!(file, address * "/Gⁿ", ts.Gⁿ)
    serializeproperty!(file, address * "/G⁻", ts.G⁻)
    return nothing
end

function serializeproperty!(file, address, ts::QuasiAdamsBashforth2TimeStepper)
    serializeproperty!(file, address * "/Gⁿ", ts.Gⁿ)
    serializeproperty!(file, address * "/G⁻", ts.G⁻)
    serializeproperty!(file, address * "/previous_Δt", ts.previous_Δt)
    return nothing
end

serializeproperty!(file, address, p::NamedTuple) = [serializeproperty!(file, address * "/$subp", getproperty(p, subp)) for subp in keys(p)]
serializeproperty!(file, address, s::StructArray) = (file[address] = replace_storage(Array, s))
serializeproperty!(file, address, p::LagrangianParticles) = serializeproperty!(file, address, p.properties)

saveproperties!(file, structure, ps) = [saveproperty!(file, "$p", getproperty(structure, p)) for p in ps]
serializeproperties!(file, structure, ps) = [serializeproperty!(file, "$p", getproperty(structure, p)) for p in ps]

# Don't check arrays because we don't need that noise.
has_reference(T, ::AbstractArray{<:Number}) = false

# This is going to be true.
has_reference(::Type{T}, ::NTuple{N, <:T}) where {N, T} = true

# Short circuit on fields.
has_reference(T::Type{Function}, f::Field) =
    has_reference(T, f.data) || has_reference(T, f.boundary_conditions)

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

""" Returns the schedule for output averaging determined by the first output value. """
function output_averaging_schedule(ow::AbstractOutputWriter)
    first_output = first(values(ow.outputs))
    return output_averaging_schedule(first_output)
end

output_averaging_schedule(output) = nothing # fallback

show_array_type(a::Type{Array{T}}) where T = "Array{$T}"

"""
    auto_extension(filename, ext)                                                             

If `filename` ends in `ext`, return `filename`. Otherwise return `filename * ext`.
"""
function auto_extension(filename, ext) 
    Next = length(ext)
    filename[end-Next+1:end] == ext || (filename *= ext)
    return filename
end
