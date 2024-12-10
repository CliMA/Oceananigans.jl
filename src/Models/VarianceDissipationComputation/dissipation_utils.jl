import Oceananigans.Utils: KernelParameters

tracer_closure_dissipation(grid, K, ::Nothing, tracer_id) = nothing
tracer_closure_dissipation(grid, K, c::Tuple,  tracer_id) = 
    Tuple(tracer_closure_dissipation(grid, K[i], c[i], tracer_id) for i in eachindex(c))

function tracer_closure_dissipation(K, c, tracer_id)
    κ = diffusivity(c, K, tracer_id)
    include_dissipation = !(κ isa Number) || (κ != 0)
    return ifelse(include_dissipation, tracer_fluxes(grid), nothing)
end

enstrophy_closure_dissipation(grid, K, c::Tuple) = 
    Tuple(enstrophy_closure_dissipation(grid, K[i], c[i]) for i in eachindex(c))

# Fallback
enstrophy_closure_dissipation(grid, K, ::Nothing) = nothing

function enstrophy_closure_dissipation(K, c)
    ν = viscosity(c, K)
    include_dissipation = !(ν isa Number) || (ν != 0)
    return ifelse(include_dissipation, vorticity_fluxes(grid), nothing)
end

@inline getadvection(advection, tracer_name) = advection
@inline getadvection(advection::NamedTuple, tracer_name) = 
    @inbounds ifelse(tracer_name == :ζ, advection.momentum, advection[tracer_name])

@inline function KernelParameters(f::Field)
    sz = size(f.data)
    of = f.data.offsets
    return KernelParameters(sz, of)
end

function tracer_fluxes(grid)
    x = XFaceField(grid)
    y = YFaceField(grid)
    z = ZFaceField(grid)

    return (; x, y, z)
end

function vorticity_fluxes(grid)
    x = YFaceField(grid)
    y = XFaceField(grid)
    
    return (; x, y)
end
