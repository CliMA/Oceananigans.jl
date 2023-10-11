module OceananigansEnzymeCoreExt

using Oceananigans

isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{Type{Field}},
                                                 ::RT,
                                                 loc::Union{EnzymeCore.Const{<:Tuple},
                                                 EnzymeCore.Duplicated{<:Tuple}},
                                                 grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid},
                                                 T::EnzymeCore.Const{<:DataType}; kw...) where RT
  primal = func.val(loc.val, grid.val, T.val; kw...)

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

  return EnzymeCore.EnzymeRules.AugmentedReturn{eltype(RT), EnzymeCore.EnzymeRules.width(config) == 1 ? eltype(RT) : NTuple{EnzymeCore.EnzymeRules.width(config), eltype(RT)}, Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::EnzymeCore.Const{Type{Field}}, ::RT, tape, loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}}, grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid}, T::EnzymeCore.Const{<:DataType}; kw...) where RT
  return (nothing, nothing, nothing, nothing)
end

end