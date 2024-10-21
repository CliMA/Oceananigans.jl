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
  AreaField(grid, loc=(Center, Center, Nothing)) = MetricField(loc, grid, Oceananigans.AbstractOperations.Az)

struct RPEDensityOperation{Z, R, V, A, B}
    z★ :: Z
    ρ :: R
    vol :: V
    A :: A
    buoyancy :: B
end

@inline _density_operation(i, j, k, grid, buoyancy, tracers) = buoyancy.equation_of_state.reference_density * 
            (1 - buoyancy_perturbationᶜᶜᶜ(i, j, k, grid, buoyancy, tracers) / buoyancy.gravitational_acceleration)

DensityOperation(grid, buoyancy, tracers) = 
    KernelFunctionOperation{Center, Center, Center}(_density_operation, grid, buoyancy, tracers)

Field(operand::RPEDensityOperation) = Field{Center, Center, Center}(operand.ρ.grid; operand)

function RPEDensityOperation(grid; tracers, buoyancy)
    vol = VolumeField(grid)
    z★  = CenterField(grid)
    A   = sum(AreaField(grid))
    ρ   = Field(DensityOperation(grid, buoyancy.model, tracers))
    return RPEDensityOperation(z★, ρ, vol, A, buoyancy)
end

RPEDensityField = Field{<:Any, <:Any, <:Any, <:RPEDensityOperation}

function compute!(ε::RPEDensityField, time=nothing)
    grid = ε.grid
    arch = architecture(grid)

    z★  = ε.operand.z★
    ρ   = ε.operand.ρ
    vol = ε.operand.vol

    compute!(ρ)

    ρ_arr = Array(interior(ρ))[:]
    v_arr = Array(interior(vol))[:]

    perm           = sortperm(ρ_arr)
    sorted_ρ_field = ρ_arr[perm]
    sorted_v_field = v_arr[perm]
    integrated_v   = cumsum(sorted_v_field)    

    sorted_ρ_field = on_architecture(arch, sorted_ρ_field)
    integrated_v   = on_architecture(arch, integrated_v)

    launch!(arch, grid, :xyz, _calculate_z★, z★, ρ, sorted_ρ_field, integrated_v)
    
    z★ ./= ε.operand.A

    set!(ε, z★ * ρ)

    return ε
end

total_RPE(RPE::RPEDensityField) = sum(RPE * RPE.operand.vol) 

@kernel function _calculate_z★(z★, b, b_sorted, integrated_v)
    i, j, k = @index(Global, NTuple)
    bl  = b[i, j, k]
    i₁  = searchsortedfirst(b_sorted, bl)
    z★[i, j, k] = integrated_v[i₁] 
end

@inline function linear_interpolate(x, y, x₀)
    i₁ = searchsortedfirst(x, x₀)
    i₂ =  searchsortedlast(x, x₀)

    @inbounds y₂ = y[i₂]
    @inbounds y₁ = y[i₁]

    @inbounds x₂ = x[i₂]
    @inbounds x₁ = x[i₁]

    if i₁ > length(x)
        return y₂
    elseif i₁ == i₂
        isnan(y₁) && @show i₁, i₂, x₁, x₂, y₁, y₂
        return 
    else
        if isnan(y₁) || isnan(y₂) || isnan(x₁) || isnan(x₂) 
            @show i₁, i₂, x₁, x₂, y₁, y₂
        end
        return (y₂ - y₁) / (x₂ - x₁) * (x₀ - x₁) + y₁
    end
end