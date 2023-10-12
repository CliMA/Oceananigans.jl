module OceananigansEnzymeCoreExt

using Oceananigans

isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing

@inline batch(::Val{1}, ::Type{T}) where T = T
@inline batch(::Val{N}, ::Type{T}) where {T,N} = NTuple{N,T}

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{Type{Field}},
                                                 ::Type{<:EnzymeCore.Annotation{RT}},
                                                 loc::Union{EnzymeCore.Const{<:Tuple},
                                                 EnzymeCore.Duplicated{<:Tuple}},
                                                 grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid},
                                                 T::EnzymeCore.Const{<:DataType}; kw...) where RT
  primal = if EnzymeRules.needs_primal(config)
      func.val(loc.val, grid.val, T.val; kw...)
  else
      nothing
  end

  if haskey(kw, :a)
    # copy zeroing
    kw[:data] = copy(kw[:data])
  end

  shadow = if EnzymeCore.EnzymeRules.width(config) == 1
  	func.val(loc.val, grid.val, T.val; kw...)
  else
  	ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
  		Base.@_inline_meta
	  	func.val(loc.val, grid.val, T.val; kw...)
  	end
  end

  return EnzymeCore.EnzymeRules.AugmentedReturn{EnzymeRules.needs_primal(config) ? RT : Nothing, batch(Val(EnzymeCore.EnzymeRules.width(config)), RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::EnzymeCore.Const{Type{Field}}, ::RT, tape, loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}}, grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid}, T::EnzymeCore.Const{<:DataType}; kw...) where RT
  return (nothing, nothing, nothing)
end

end
