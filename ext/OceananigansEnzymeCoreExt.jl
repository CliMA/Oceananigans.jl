module OceananigansEnzymeCoreExt

using Oceananigans
using KernelAbstractions

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
  primal = if EnzymeCore.EnzymeRules.needs_primal(config)
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

  return EnzymeCore.EnzymeRules.AugmentedReturn{EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing, batch(Val(EnzymeCore.EnzymeRules.width(config)), RT), Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1}, func::EnzymeCore.Const{Type{Field}}, ::RT, tape, loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}}, grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid}, T::EnzymeCore.Const{<:DataType}; kw...) where RT
  return (nothing, nothing, nothing)
end


function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{typeof(Oceananigans.Utils.launch!)},
                                                 ::Type{EnzymeCore.Const{Nothing}},
                                                 arch,
                                                 grid,
                                                 workspec,
                                                 kernel!,
                                                 kernel_args...;
                                                 include_right_boundaries = false,
                                                 reduced_dimensions = (),
                                                 location = nothing,
                                                 only_active_cells = nothing,
                                                 kwargs...)


    workgroup, worksize = Oceananigans.Utils.work_layout(grid.val, workspec.val;
                                      include_right_boundaries,
                                      reduced_dimensions,
                                      location)

    offset = Oceananigans.Utils.offsets(workspec.val)

    if !isnothing(only_active_cells) 
        workgroup, worksize = Oceananigans.Utils.active_cells_work_layout(workgroup, worksize, only_active_cells, grid.val) 
        offset = nothing
    end

    if worksize != 0
      
      # We can only launch offset kernels with Static sizes!!!!

      if isnothing(offset)
          loop! = kernel!.val(Oceananigans.Architectures.device(arch.val), workgroup, worksize)
          dloop! = (typeof(kernel!) <: EnzymeCore.Const) ? nothing : kernel!.dval(Oceananigans.Architectures.device(arch.val), workgroup, worksize)
      else
          loop! = kernel!.val(Oceananigans.Architectures.device(arch.val), KernelAbstractions.StaticSize(workgroup), Oceananigans.Utils.OffsetStaticSize(contiguousrange(worksize, offset))) 
          dloop! = (typeof(kernel!) <: EnzymeCore.Const) ? nothing : kernel!.val(Oceananigans.Architectures.device(arch.val), KernelAbstractions.StaticSize(workgroup), Oceananigans.Utils.OffsetStaticSize(contiguousrange(worksize, offset)))
      end

      @debug "Launching kernel $kernel! with worksize $worksize and offsets $offset from $workspec.val"


      duploop = (typeof(kernel!) <: EnzymeCore.Const) ? EnzymeCore.Const(loop!) : EnzymeCore.Duplicated(loop!, dloop!)

      config2 = EnzymeCore.EnzymeRules.Config{#=needsprimal=#false, #=needsshadow=#false, #=width=#EnzymeCore.EnzymeRules.width(config), EnzymeCore.EnzymeRules.overwritten(config)[5:end]}()
      subtape = EnzymeCore.EnzymeRules.augmented_primal(config2, duploop, EnzymeCore.Const{Nothing}, kernel_args...).tape

      tape = (duploop, subtape)
    else
      tape = nothing
    end

    return EnzymeCore.EnzymeRules.AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, tape)
end

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
                                                func::EnzymeCore.Const{typeof(Oceananigans.Utils.launch!)},
                                                 ::Type{EnzymeCore.Const{Nothing}},
                                                 tape,
                                                 arch,
                                                 grid,
                                                 workspec,
                                                 kernel!,
                                                 kernel_args...;
                                                 include_right_boundaries = false,
                                                 reduced_dimensions = (),
                                                 location = nothing,
                                                 only_active_cells = nothing,
                                                 kwargs...)

  subrets = if tape !== nothing
    duploop, subtape = tape

    config2 = EnzymeCore.EnzymeRules.Config{#=needsprimal=#false, #=needsshadow=#false, #=width=#EnzymeCore.EnzymeRules.width(config), EnzymeCore.EnzymeRules.overwritten(config)[5:end]}()

    EnzymeCore.EnzymeRules.reverse(config2, duploop, EnzymeCore.Const{Nothing}, subtape, kernel_args...)
  else
    ntuple(Val(length(kernel_args))) do _
      Base.@_inline_meta
      nothing
    end
  end

  return (nothing, nothing, nothing, nothing, subrets...)

end

end
