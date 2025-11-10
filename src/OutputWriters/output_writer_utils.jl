using StructArrays: StructArray, replace_storage
using Oceananigans.Grids: on_architecture, architecture
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedGrid, Partition
using Oceananigans.Fields: AbstractField, indices, boundary_conditions, instantiated_location, ConstantField, ZeroField, OneField
using Oceananigans.BoundaryConditions: bc_str, FieldBoundaryConditions, ContinuousBoundaryFunction, DiscreteBoundaryFunction
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, RungeKutta3TimeStepper
using Oceananigans.Utils: AbstractSchedule
using Oceananigans.OutputReaders: auto_extension

#####
##### Output writer utilities
#####

struct NoFileSplitting end
(::NoFileSplitting)(model) = false
Base.summary(::NoFileSplitting) = "NoFileSplitting"
Base.show(io::IO, nfs::NoFileSplitting) = print(io, summary(nfs))
initialize!(::NoFileSplitting, model) = nothing

mutable struct FileSizeLimit <: AbstractSchedule
    size_limit :: Float64
    path :: String
end

"""
    FileSizeLimit(size_limit [, path=""])

Return a schedule that actuates when the file at `path` exceeds
the `size_limit`.

The `path` is automatically added and updated when `FileSizeLimit` is
used with an output writer, and should not be provided manually.
"""
FileSizeLimit(size_limit) = FileSizeLimit(size_limit, "")
(fsl::FileSizeLimit)(model) = filesize(fsl.path) ≥ fsl.size_limit

function Base.summary(fsl::FileSizeLimit)
    current_size_str = pretty_filesize(filesize(fsl.path))
    size_limit_str = pretty_filesize(fsl.size_limit)
    return string("FileSizeLimit(size_limit=", size_limit_str,
                              ", path=", fsl.path, " (", current_size_str, ")")
end

Base.show(io::IO, fsl::FileSizeLimit) = print(io, summary(fsl))

# Update schedule based on user input
update_file_splitting_schedule!(schedule, filepath) = nothing

function update_file_splitting_schedule!(schedule::FileSizeLimit, filepath)
    schedule.path = filepath
    return nothing
end

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

const ConstantFields = Union{ConstantField, ZeroField, OneField}

# Some specific things
saveproperty!(file, address, p::Union{Number, Array}) = file[address] = p
saveproperty!(file, address, p::ConstantFields)       = file[address] = p
saveproperty!(file, address, p::AbstractRange)        = file[address] = collect(p)
saveproperty!(file, address, p::AbstractArray)        = file[address] = Array(parent(p))
saveproperty!(file, address, p::Function)             = nothing
saveproperty!(file, address, p::Tuple)                = [saveproperty!(file, address * "/$i", p[i]) for i in 1:length(p)]
saveproperty!(file, address, grid::AbstractGrid)      = _saveproperty!(file, address, on_architecture(CPU(), grid))

function saveproperty!(file, address, grid::DistributedGrid)
    arch = architecture(grid)
    cpu_arch = Distributed(CPU(); partition = Partition(arch.ranks...))
    _saveproperty!(file, address, on_architecture(cpu_arch, grid))
end

# Special saveproperty! so boundary conditions are easily readable outside julia.
function saveproperty!(file, address, bcs::FieldBoundaryConditions)
    for boundary in propertynames(bcs)
        if !(boundary == :kernels || boundary == :ordered_bcs) # Skip kernels and ordered_bcs.
            bc = getproperty(bcs, boundary)
            file[address * "/$boundary/type"] = bc_str(bc)

            if bc === nothing || bc.condition isa Function || bc.condition isa ContinuousBoundaryFunction
                file[address * "/$boundary/condition"] = missing
            else
                file[address * "/$boundary/condition"] = on_architecture(CPU(), bc.condition)
            end
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

function serializeproperty!(file, address, grid::DistributedGrid)
    arch = architecture(grid)
    cpu_arch = Distributed(CPU(); partition = arch.partition)
    file[address] = on_architecture(cpu_arch, grid)
end

function remove_function_bcs(fbcs::FieldBoundaryConditions)
    west     = has_reference(Function, fbcs.west)     ? missing : fbcs.west
    east     = has_reference(Function, fbcs.east)     ? missing : fbcs.east
    south    = has_reference(Function, fbcs.south)    ? missing : fbcs.south
    north    = has_reference(Function, fbcs.north)    ? missing : fbcs.north
    bottom   = has_reference(Function, fbcs.bottom)   ? missing : fbcs.bottom
    top      = has_reference(Function, fbcs.top)      ? missing : fbcs.top
    immersed = has_reference(Function, fbcs.immersed) ? missing : fbcs.immersed
    new_fbcs = FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
    return new_fbcs
end

function serializeproperty!(file, address, fbcs::FieldBoundaryConditions)
    new_fbcs = remove_function_bcs(fbcs)
    file[address] = on_architecture(CPU(), new_fbcs)
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
    return nothing
end

serializeproperty!(file, address, p::NamedTuple) = [serializeproperty!(file, address * "/$subp", getproperty(p, subp)) for subp in keys(p)]
serializeproperty!(file, address, s::StructArray) = (file[address] = replace_storage(Array, s))

saveproperties!(file, structure, ps) = [saveproperty!(file, "$p", getproperty(structure, p)) for p in ps]
serializeproperties!(file, structure, ps) = [serializeproperty!(file, "$p", getproperty(structure, p)) for p in ps]
serializeproperties!(file, structure, ps, addr) = [serializeproperty!(file, "$addr/$p", getproperty(structure, p)) for p in ps]

# Don't check arrays because we don't need that noise.
has_reference(T, ::AbstractArray{<:Number}) = false

# This is going to be true.
has_reference(::Type{T}, ::NTuple{N, <:T}) where {N, T} = true

# Short circuit on fields.
has_reference(T::Type{Function}, f::Field) =
    has_reference(T, f.data) || has_reference(T, f.boundary_conditions)

# Short circuit on boundary conditions.
has_reference(T::Type{Function}, bcs::FieldBoundaryConditions) =
    has_reference(T, bcs.west) || has_reference(T, bcs.east) ||
    has_reference(T, bcs.south) || has_reference(T, bcs.north) ||
    has_reference(T, bcs.bottom) || has_reference(T, bcs.top) ||
    has_reference(T, bcs.immersed)

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

#####
##### Architecture suffix
#####

with_architecture_suffix(arch, filename, ext) = filename

function with_architecture_suffix(arch::Distributed, filename, ext)
    Ne = length(ext)
    prefix = filename[1:end-Ne]
    rank = arch.local_rank
    prefix *= "_rank$rank"
    return prefix * ext
end

