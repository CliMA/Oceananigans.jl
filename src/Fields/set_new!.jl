using CUDA
using KernelAbstractions: @kernel, @index, CUDADevice
using Oceananigans.Architectures: device, GPU
using Oceananigans.Utils: work_layout

function set_new!(model; kwargs...)
    for (fldname, value) in kwargs
        if fldname ∈ propertynames(model.transports)
            ϕ = getproperty(model.transports, fldname)
        elseif fldname ∈ propertynames(model.heights)
            ϕ = getproperty(model.heights, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.transports, model.heights or model.tracers."))
        end
        set_new!(ϕ, value)
    end
    return nothing
end

function set_new!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set_new!(ϕ, value)
    end
    return nothing
end

set_new!(uh::AbstractField, vh::Number) = @. uh.data.parent = vh

set_new!(uh::AbstractField{X, Y, A}, vh::AbstractField{X, Y, A}) where {X, Y, A} =
    @. uh.data.parent = vh.data.parent

# Niceties
const AbstractCPUField =
    AbstractField{X, Y, A, G} where {X, Y, A<:OffsetArray{T, D, <:Array} where {T, D}, G}

const AbstractReducedCPUField =
    AbstractReducedField{X, Y, A, G} where {X, Y, A<:OffsetArray{T, D, <:Array} where {T, D}, G}

"Set the CPU field `uh` to the array `vh`."
function set_new!(uh::AbstractCPUField, vh::Array)

    Sx, Sy, Sz = size(uh)
    for k in 1:Sz, j in 1:Sy, i in 1:Sx
        uh[i, j, k] = vh[i, j, k]
    end

    return nothing
end

""" Returns an AbstractReducedField on the CPU. """
function similar_cpu_field(uh::AbstractReducedField)
    FieldType = typeof(uh).name.wrapper
    return FieldType(location(uh), CPU(), uh.grid; dims=uh.dims)
end

""" Set the CPU field `uh` data to the function `f(x, y)`. """
set_new!(uh::AbstractCPUField, f::Function) = interior(uh) .= f.(nodes(u; reshape=true)...)

#####
##### set! for fields on the GPU
#####

@hascuda begin
    const AbstractGPUField =
        AbstractField{X, Y, A, G} where {X, Y, A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}

    const AbstractReducedGPUField =
        AbstractReducedField{X, Y, A, G} where {X, Y, A<:OffsetArray{T, D, <:CuArray} where {T, D}, G}

    """ Returns a field on the CPU with `nothing` boundary conditions. """
    function similar_cpu_field(uh)
        FieldType = typeof(uh).name.wrapper
        return FieldType(location(uh), CPU(), uh.grid, nothing)
    end

    """ Set the GPU field `uh` to the array `vh`. """
    function set_new!(uh::AbstractGPUField, vh::Array)
        vh_field = similar_cpu_field(uh)
    
        set_new!(vh_field, vh)
        set_new!(uh, vh_field)
    
        return nothing
    end

    #FJP: does set_gpu! have to be changed?
    """ Set the GPU field `uh` to the CuArray `vh`. """
    function set_new!(uh::AbstractGPUField, vh::CuArray)
    
        launch!(GPU(), uh.grid, :xy, _set_gpu!, uh.data, vh, uh.grid,
                include_right_boundaries=true, location=location(uh))
    
        return nothing
    end
    
    @kernel function _set_gpu!(uh, vh, grid)
        i, j, k = @index(Global, NTuple)
        @inbounds u[i, j, k] = v[i, j, k]
    end

    """ Set the CPU field `uh` data to the GPU field data of `vh`. """
    set_new!(uh::AbstractCPUField, vh::AbstractGPUField) = uh.data.parent .= Array(vh.data.parent)
    
    """ Set the GPU field `uh` data to the CPU field data of `vh`. """
    set_new!(uh::AbstractGPUField, vh::AbstractCPUField) = copyto!(uh.data.parent, vh.data.parent)
    
    """ Set the GPU field `uh` data to the function `f(x, y)`. """
    function set_new!(uh::AbstractGPUField, f::Function)
        # Create a temporary field with bcs = nothing.
        vh_field = similar_cpu_field(uh)
    
        set_new!(vh_field, f)
        set_new!(uh, vh_field)
    
        return nothing
    end
end
