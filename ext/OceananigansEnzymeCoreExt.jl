module OceananigansEnzymeCoreExt

using Oceananigans

isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing

function EnzymeCore.EnzymeRules.augmented_primal(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
                                                 func::Const{Type{Field}},
                                                 ::Type{Duplicated{RT}},
                                                 loc::Union{Const{<:Tuple},
                                                 Duplicated{<:Tuple}},
                                                 grid::Const{<:Oceananigans.Grids.AbstractGrid},
                                                 T::Const{<:DataType}; kw...) where RT
  primal = func.val(loc.val, grid.val, T.val; kw...)

  if haskey(kw, :a)
    # copy zeroing
    kw[:data] = copy(kw[:data])
  end

  shadow = func.val(loc.val, grid.val, T.val; kw...)

  return EnzymeCore.EnzymeRules.AugmentedReturn{RT, RT, Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::Const{Type{Field}}, ::Type{Duplicated{RT}}, tape, loc::Union{Const{<:Tuple}, Duplicated{<:Tuple}}, grid::Const{<:Oceananigans.Grids.AbstractGrid}, T::Const{<:DataType}; kw...) where RT
  return (nothing, nothing, nothing, nothing)
end

end