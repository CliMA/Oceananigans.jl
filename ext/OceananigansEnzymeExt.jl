module OceananigansEnzymeExt

using Oceananigans
using Oceananigans.Fields: FunctionField
using Oceananigans.Utils: contiguousrange
using KernelAbstractions

#isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

using Enzyme
using Enzyme: EnzymeCore
using EnzymeCore: Active

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing

@inline batch(::Val{1}, ::Type{T}) where T = T
@inline batch(::Val{N}, ::Type{T}) where {T, N} = NTuple{N, T}

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

    P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
    B = batch(Val(EnzymeCore.EnzymeRules.width(config)), RT)
    return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, Nothing}(primal, shadow, nothing)
end

#####
##### Field
#####

function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
                                        func::EnzymeCore.Const{Type{Field}},
                                        ::RT,
                                        tape,
                                        loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}},
                                        grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid},
                                        T::EnzymeCore.Const{<:DataType}; kw...) where RT
    return (nothing, nothing, nothing)
end

#####
##### FunctionField
#####

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 enzyme_func::EnzymeCore.Const{<:Type{<:FunctionField}},
                                                 ::Type{<:EnzymeCore.Annotation{RT}},
                                                 function_field_func,
                                                 grid;
                                                 clock = nothing,
                                                 parameters = nothing) where RT
    FunctionFieldType = enzyme_func.val     

    primal = if EnzymeCore.EnzymeRules.needs_primal(config)
        FunctionFieldType(function_field_func.val, grid.val; clock, parameters)
    else
        nothing
    end

    # function_field_func can be Active, Const (inactive), Duplicated (active but mutable)
    function_field_is_active = function_field_func isa Active

    # Support batched differentiation!
    config_width = EnzymeCore.EnzymeRules.width(config)

    dactives = if function_field_is_active
        runtime_func_type = Core.Typeof(function_field_func.val)

        if config_width == 1
            Ref(Enzyme.Compiler.make_zero(runtime_func_type,
                                          IdDict(),
                                          function_field_func.val))
        else
            ntuple(Val(config_width)) do i
                Base.@_inline_meta
                Ref(Enzyme.Compiler.make_zero(runtime_func_type,
                                              IdDict(),
                                              function_field_func.val))
            end
        end
    else
        nothing
    end

    shadow = if config_width == 1
        dfunction_field_func = if function_field_is_active
            dactives[]
        else
            function_field_func.dval
        end

        FunctionFieldType(dfunction_field_func, grid.val; clock, parameters)
    else
  	    ntuple(Val(config_width)) do i
  		    Base.@_inline_meta

            dfunction_field_func = if function_field_is_active
                dactives[i][]
            else
                function_field_func.dval[i]
            end

            FunctionFieldType(dfunction_field_func, grid.val; clock, parameters)
  	    end
    end

    P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
    B = batch(Val(EnzymeCore.EnzymeRules.width(config)), RT)
    D = typeof(dactives)

    return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, D}(primal, shadow, dactives)
end

function EnzymeCore.EnzymeRules.reverse(config,
                                        enzyme_func::EnzymeCore.Const{<:Type{<:FunctionField}},
                                        ::RT,
                                        tape,
                                        function_field_func,
                                        grid;
                                        clock = nothing,
                                        parameters = nothing) where RT

    dactives = if function_field_func isa Active
        if Enzyme.Core.EnzymeRules.width(config) == 1
            tape[]
        else
            ntuple(Val(Enzyme.Core.EnzymeRules.width(config))) do i
                Base.@_inline_meta
                tape[i][]
            end
        end
    else
        nothing
    end

    # return (dactives, grid (nothing))
    return (dactives, nothing)
end

#####
##### launch!
#####

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
