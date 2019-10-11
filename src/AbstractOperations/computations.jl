struct Computation{RT, OU, OP, G}
      operation :: OP
         result :: OU
           grid :: G
    return_type :: RT
end

function Computation(op, result; return_type=Array)
    return Computation(op, result, op.grid, return_type)
end

architecture(comp::Computation) = architecture(comp.result)
Base.parent(comp::Computation) = comp

function compute!(comp::Computation)
    arch = architecture(comp)
    @launch device(arch) config=launch_config(comp.grid, 3) _compute!(data(comp.result), comp.grid, 
                                                                      comp.operation)
    return nothing
end

function _compute!(result, grid, operation)
    @loop for k in (1:grid.Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds result[i, j, k] = operation[i, j, k]
            end
        end
    end
    return nothing
end

function (comp::Computation)(args...)
    compute!(comp)
    return comp.return_type(data(comp.result))
end

function (comp::Computation{<:Nothing})(args...)
    compute!(comp)
    return comp.result
end

#####
##### Functionality for using computations with HorizontalAverage
#####

function HorizontalAverage(op::AbstractOperation, result; kwargs...)
    computation = Computation(op, result)
    return HorizontalAverage(computation; kwargs...)
end

HorizontalAverage(op::AbstractOperation, model::AbstractModel; kwargs...) = 
    HorizontalAverage(op, model.pressures.pHY′; kwargs...)

function run_diagnostic(model, havg::HorizontalAverage{<:Computation})
    compute!(havg.field)
    zero_halo_regions!(parent(havg.field.result), model.grid)
    sum!(havg.result, parent(havg.field.result))
    normalize_horizontal_sum!(havg, model.grid)
    return nothing
end
