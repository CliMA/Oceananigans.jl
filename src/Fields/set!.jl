using CUDA
using KernelAbstractions: @kernel, @index
using Adapt: adapt_structure

using Oceananigans.Grids: on_architecture
using Oceananigans.Architectures: device, GPU, CPU, MetalBackend
using Oceananigans.Utils: work_layout

function set!(Φ::NamedTuple; kwargs...)
    for (fldname, value) in kwargs
        ϕ = getproperty(Φ, fldname)
        set!(ϕ, value)
    end
    return nothing
end

function set!(u::Field, v)
    u .= v # fallback
    return u
end

function set!(u::Field, f::Function)
    if architecture(u) isa GPU || architecture(u) isa MetalBackend
        cpu_grid = on_architecture(CPU(), u.grid)
        u_cpu = Field(location(u), cpu_grid; indices = indices(u))
        f_field = field(location(u), f, cpu_grid)
        set!(u_cpu, f_field)
        set!(u, u_cpu)
    elseif architecture(u) isa CPU
        f_field = field(location(u), f, u.grid)
        set!(u, f_field)
    end

    return u
end

function set!(u::Field, f::Union{Array, CuArray, OffsetArray})
    f = arch_array(architecture(u), f)
    u .= f
    return u
end

function set!(u::Field, v::Field)
    # We implement some niceities in here that attempt to copy halo data,
    # and revert to copying just interior points if that fails.
    
    if architecture(u) === architecture(v)
        # Note: we could try to copy first halo point even when halo
        # regions are a different size. That's a bit more complicated than
        # the below so we leave it for the future.
        
        try # to copy halo regions along with interior data
            parent(u) .= parent(v)
        catch # this could fail if the halo regions are different sizes?
            # copy just the interior data
            interior(u) .= interior(v)
        end
    else
        v_data = arch_array(architecture(u), v.data)
        
        # As above, we permit ourselves a little ambition and try to copy halo data:
        try
            parent(u) .= parent(v_data)
        catch
            interior(u) .= interior(v_data, location(v), v.grid, v.indices)
        end
    end

    return u
end
