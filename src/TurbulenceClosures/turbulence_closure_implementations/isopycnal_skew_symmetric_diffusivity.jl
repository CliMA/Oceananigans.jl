struct IsopycnalSkewSymmetricDiffusivity{K, S, M, L} <: AbstractTurbulenceClosure{ExplicitTimeDiscretization}
                    κ_skew :: K
               κ_symmetric :: S
          isopycnal_tensor :: M
             slope_limiter :: L
    
    function IsopycnalSkewSymmetricDiffusivity(κ_skew::K, κ_symmetric::S, isopycnal_tensor::I, slope_limiter::L) where {K, S, I, L}

        isopycnal_tensor isa SmallSlopeIsopycnalTensor ||
            error("Only isopycnal_tensor=SmallSlopeIsopycnalTensor() is currently supported.")

        return new{K, S, I, L}(κ_skew, κ_symmetric, isopycnal_tensor, slope_limiter)
    end
end

const ISSD = IsopycnalSkewSymmetricDiffusivity
const ISSDVector = AbstractVector{<:ISSD}
const FlavorOfISSD = Union{ISSD, ISSDVector}
const issd_coefficient_loc = (Center, Center, Center)


"""
    IsopycnalSkewSymmetricDiffusivity([FT=Float64;]
                                      κ_skew = 0,
                                      κ_symmetric = 0,
                                      isopycnal_tensor = SmallSlopeIsopycnalTensor(),
                                      slope_limiter = nothing)

Return parameters for an isopycnal skew-symmetric tracer diffusivity with skew diffusivity
`κ_skew` and symmetric diffusivity `κ_symmetric` that uses an `isopycnal_tensor` model for
for calculating the isopycnal slopes, and (optionally) applying a `slope_limiter` to the
calculated isopycnal slope values.
    
Both `κ_skew` and `κ_symmetric` may be constants, arrays, fields, or functions of `(x, y, z, t)`.
"""
IsopycnalSkewSymmetricDiffusivity(FT=Float64; κ_skew=0, κ_symmetric=0, isopycnal_tensor=SmallSlopeIsopycnalTensor(), slope_limiter=nothing) =
    IsopycnalSkewSymmetricDiffusivity(convert_diffusivity(FT, κ_skew), convert_diffusivity(FT, κ_symmetric), isopycnal_tensor, slope_limiter)

function with_tracers(tracers, closure::ISSD)
    κ_skew = !isa(closure.κ_skew, NamedTuple) ? closure.κ_skew : tracer_diffusivities(tracers, closure.κ_skew)
    κ_symmetric = !isa(closure.κ_symmetric, NamedTuple) ? closure.κ_symmetric : tracer_diffusivities(tracers, closure.κ_symmetric)
    return IsopycnalSkewSymmetricDiffusivity(κ_skew, κ_symmetric, closure.isopycnal_tensor, closure.slope_limiter)
end

# For ensembles of closures
function with_tracers(tracers, closure_vector::ISSDVector)
    arch = architecture(closure_vector)

    if arch isa Architectures.GPU
        closure_vector = Vector(closure_vector)
    end

    Ex = length(closure_vector)
    closure_vector = [with_tracers(tracers, closure_vector[i]) for i=1:Ex]

    return arch_array(arch, closure_vector)
end

@inline get_tracer_κ(κ::NamedTuple, tracer_index) = @inbounds κ[tracer_index]
@inline get_tracer_κ(κ, tracer_index) = κ

# Interface
@inline skew_diffusivity(closure::ISSD, id, K) = get_tracer_κ(closure.κ_skew, id)
@inline symmetric_diffusivity(closure::ISSD, id, K) = get_tracer_κ(closure.κ_symmetric, id)
@inline flux_tapering(closure::ISSD) = closure.slope_limiter
@inline isopycnal_tensor(closure::ISSD) = closure.isopycnal_tensor

calculate_diffusivities!(diffusivity_fields, closure::FlavorOfISSD, model) = nothing

#####
##### Show
#####

Base.show(io::IO, closure::ISSD) =
    print(io, "IsopycnalSkewSymmetricDiffusivity: " *
              "(κ_symmetric=$(closure.κ_symmetric), κ_skew=$(closure.κ_skew), " *
              "(isopycnal_tensor=$(closure.isopycnal_tensor), slope_limiter=$(closure.slope_limiter))")
              
