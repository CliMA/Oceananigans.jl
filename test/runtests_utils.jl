using KernelAbstractions

#####
##### Useful utilities
#####

function get_model_field(field_name, model)
    if field_name ∈ (:u, :v, :w)
        return getfield(model.velocities, field_name)
    else
        return getfield(model.tracers, field_name)
    end
end

datatuple(A) = NamedTuple{propertynames(A)}(Array(data(a)) for a in A)

function get_output_tuple(output, iter, tuplename)
    file = jldopen(output.filepath, "r")
    output_tuple = file["timeseries/$tuplename/$iter"]
    close(file)
    return output_tuple
end

#####
##### Useful kernels
#####

@kernel function ∇²!(grid, f, ∇²f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end
