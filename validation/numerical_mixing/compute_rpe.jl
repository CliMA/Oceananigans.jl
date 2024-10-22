using Oceananigans.AbstractOperations: GridMetricOperation
using Oceananigans.Grids: architecture, znode
using Oceananigans.BuoyancyModels: buoyancy_perturbationᶜᶜᶜ
using Oceananigans.Architectures: device, on_architecture
using Oceananigans.Utils: launch!
using KernelAbstractions: @index, @kernel

import Oceananigans.AbstractOperations: compute!
import Oceananigans.Fields: Field

MetricField(loc, grid, metric) = compute!(Field(GridMetricOperation(loc, metric, grid)))

VolumeField(grid, loc=(Center, Center, Center)) = MetricField(loc, grid, Oceananigans.AbstractOperations.volume)
AreaField(grid, loc=(Center, Center, Nothing))  = MetricField(loc, grid, Oceananigans.AbstractOperations.Az)

struct RPEDensityOperation{Z, R, V, A, B}
    z★ :: Z
    b :: R
    vol :: V
    A :: A
    buoyancy :: B
end

BuoyancyOperation(grid, buoyancy, tracers) = 
    KernelFunctionOperation{Center, Center, Center}(buoyancy_perturbationᶜᶜᶜ, grid, buoyancy, tracers)

Field(operand::RPEDensityOperation) = Field{Center, Center, Center}(operand.b.grid; operand)

function RPEDensityOperation(grid; tracers, buoyancy)
    vol = VolumeField(grid)
    z★  = CenterField(grid)
    A   = sum(AreaField(grid))
    b   = Field(BuoyancyOperation(grid, buoyancy.model, tracers))
    return RPEDensityOperation(z★, b, vol, A, buoyancy)
end

RPEDensityField = Field{<:Any, <:Any, <:Any, <:RPEDensityOperation}

function compute!(ε::RPEDensityField, time=nothing)
    grid = ε.grid
    arch = architecture(grid)

    z★  = ε.operand.z★
    b   = ε.operand.b
    vol = ε.operand.vol

    compute!(b)

    b_arr = Array(interior(b))[:]
    v_arr = Array(interior(vol))[:]

    perm           = sortperm(b_arr)
    sorted_b_field = b_arr[perm]
    sorted_v_field = v_arr[perm]
    integrated_v   = cumsum(sorted_v_field)    

    sorted_b_field = on_architecture(arch, sorted_b_field)
    integrated_v   = on_architecture(arch, integrated_v)

    launch!(arch, grid, :xyz, _calculate_z★, z★, b, sorted_b_field, integrated_v, ε.operand.A)
    set!(ε, z★ * (1 - b / 9.80665) * 1020)

    return ε
end

total_RPE(RPE::RPEDensityField) = sum(RPE * RPE.operand.vol) 

function total_RPE(RPE::Field) 
    grid = RPE.grid
    vol  = VolumeField(grid)
    return sum(RPE * vol)
end

@kernel function _calculate_z★(z★, b, b_sorted, integrated_v, A)
    i, j, k = @index(Global, NTuple)
    bl  = b[i, j, k]
    i₁  = searchsortedlast(b_sorted, bl)
    i₂  = searchsortedfirst(b_sorted, bl)
    z★₁ = integrated_v[i₁] / A
    z★₂ = integrated_v[i₂] / A

    @inbounds z★[i, j, k] = (z★₁ + z★₂) / 2
end