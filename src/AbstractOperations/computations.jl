struct Computation{OU, OP, G, RT}
       operator :: OP
         result :: OU
           grid :: G
    return_type :: RT
    function Computation(op, result, return_type)
        return new{typeof(op), typeof(result), typeof(op.grid), 
                   typeof(return_type)}(op, result, op.grid, return_type)
    end
end

function compute!(comp::Computation)
    arch = architecture(comp.result)
    @launch device(arch) config=launch_config(grid, 3) _compute!(comp.result, comp.grid, comp.operator)
    return nothing
end

function _compute!(result, grid, operator)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds result[i, j, k] = operator[i, j, k]
            end
        end
    end
    return nothing
end

function (comp::Computation)(args...)
    compute!(comp)
    return comp.return_type(comp.result)
end
