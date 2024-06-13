module OceananigansEnzymeExt

using Oceananigans
using Oceananigans.Fields: FunctionField
using Oceananigans.Utils: contiguousrange
#Oceananigans.Models.HydrostaticFreeSurfaceModels.top_tracer_boundary_conditions

using KernelAbstractions

#isdefined(Base, :get_extension) ? (import EnzymeCore) : (import ..EnzymeCore)
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

using Enzyme: EnzymeCore
using Enzyme.EnzymeCore: Active, Const, Duplicated

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.BoundaryConditions.parent_size_and_offset), x...) = nothing
@inline EnzymeCore.EnzymeRules.inactive_type(v::Type{Oceananigans.Utils.KernelParameters}) = true

@inline batch(::Val{1}, ::Type{T}) where T = T
@inline batch(::Val{N}, ::Type{T}) where {T, N} = NTuple{N, T}

# function EnzymeCore.EnzymeRules.augmented_primal(config,
#                                                  func::EnzymeCore.Const{Type{Field}},
#                                                  ::Type{<:EnzymeCore.Annotation{RT}},
#                                                  loc::Union{EnzymeCore.Const{<:Tuple},
#                                                  EnzymeCore.Duplicated{<:Tuple}},
#                                                  grid::EnzymeCore.Annotation{<:Oceananigans.Grids.AbstractGrid},
#                                                  T::EnzymeCore.Const{<:DataType}; kw...) where RT
# 
#     primal = if EnzymeCore.EnzymeRules.needs_primal(config)
#         func.val(loc.val, grid.val, T.val; kw...)
#     else
#         nothing
#     end
# 
#     if haskey(kw, :a)
#         # copy zeroing
#         kw[:data] = copy(kw[:data])
#     end
# 
#     shadow = if EnzymeCore.EnzymeRules.width(config) == 1
#         func.val(loc.val, grid.val, T.val; kw...)
#     else
#         ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
#             Base.@_inline_meta
#             func.val(loc.val, grid.val, T.val; kw...)
#         end
#     end
# 
#     P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
#     B = batch(Val(EnzymeCore.EnzymeRules.width(config)), RT)
#     return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, Nothing}(primal, shadow, nothing)
# end
# 
# #####
# ##### Field
# #####
# 
# function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
#                                         func::EnzymeCore.Const{Type{Field}},
#                                         ::RT,
#                                         tape,
#                                         loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}},
#                                         grid::EnzymeCore.Const{<:Oceananigans.Grids.AbstractGrid},
#                                         T::EnzymeCore.Const{<:DataType}; kw...) where RT
#     return (nothing, nothing, nothing)
# end
# 
# function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
#                                         func::EnzymeCore.Const{Type{Field}},
#                                         ::RT,
#                                         tape,
#                                         loc::Union{EnzymeCore.Const{<:Tuple}, EnzymeCore.Duplicated{<:Tuple}},
#                                         grid::EnzymeCore.Active{<:Oceananigans.Grids.AbstractGrid},
#                                         T::EnzymeCore.Const{<:DataType}; kw...) where RT
#     return (nothing, EnzymeCore.make_ero(grid), nothing)
# end

#####
##### FunctionField
#####

# @inline FunctionField(L::Tuple, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)

# function EnzymeCore.EnzymeRules.augmented_primal(config,
#                                                  enzyme_func::Union{EnzymeCore.Const{<:Type{<:FunctionField}}, EnzymeCore.Const{Type{FT2}}},
#                                                  ::Type{<:EnzymeCore.Annotation{RT}},
#                                                  function_field_func,
#                                                  grid;
#                                                  clock = nothing,
#                                                  parameters = nothing) where {RT, FT2 <: FunctionField}
# 
#     FunctionFieldType = enzyme_func.val     
# 
#     primal = if EnzymeCore.EnzymeRules.needs_primal(config)
#         FunctionFieldType(function_field_func.val, grid.val; clock, parameters)
#     else
#         nothing
#     end
# 
#     # function_field_func can be Active, Const (inactive), Duplicated (active but mutable)
#     function_field_is_active = function_field_func isa Active
#     # @show function_field_func
# 
#     # Support batched differentiation!
#     config_width = EnzymeCore.EnzymeRules.width(config)
# 
#     dactives = if function_field_is_active
#         if config_width == 1
#             Ref(EnzymeCore.make_zero(function_field_func.val))
#         else
#             ntuple(Val(config_width)) do i
#                 Base.@_inline_meta
#                 Ref(EnzymeCore.make_zero(function_field_func.val))
#             end
#         end
#     else
#         nothing
#     end
# 
#     shadow = if config_width == 1
#         dfunction_field_func = if function_field_is_active
#             dactives[]
#         else
#             function_field_func.dval
#         end
# 
#         FunctionFieldType(dfunction_field_func, grid.val; clock, parameters)
#     else
#   	    ntuple(Val(config_width)) do i
#   		    Base.@_inline_meta
# 
#             dfunction_field_func = if function_field_is_active
#                 dactives[i][]
#             else
#                 function_field_func.dval[i]
#             end
# 
#             FunctionFieldType(dfunction_field_func, grid.val; clock, parameters)
#   	    end
#     end
# 
#     P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
#     B = batch(Val(EnzymeCore.EnzymeRules.width(config)), RT)
#     D = typeof(dactives)
# 
#     return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, D}(primal, shadow, dactives)
# end
# 
# function EnzymeCore.EnzymeRules.reverse(config,
#                                         enzyme_func::Union{EnzymeCore.Const{<:Type{<:FunctionField}}, EnzymeCore.Const{Type{FT2}}},
#                                         ::RT,
#                                         tape,
#                                         function_field_func,
#                                         grid;
#                                         clock = nothing,
#                                         parameters = nothing) where {RT, FT2 <: FunctionField}
# 
#     dactives = if function_field_func isa Active
#         if EnzymeCore.EnzymeRules.width(config) == 1
#             tape[]
#         else
#             ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
#                 Base.@_inline_meta
#                 tape[i][]
#             end
#         end
#     else
#         nothing
#     end
# 
#     # return (dactives, grid (nothing))
#     return (dactives, nothing)
# end

#####
##### launch!
#####

# function EnzymeCore.EnzymeRules.augmented_primal(config,
#                                                  func::EnzymeCore.Const{typeof(Oceananigans.Models.flattened_unique_values)},
#                                                  ::Type{<:EnzymeCore.Annotation{RT}},
#                                                  a) where RT
# 
#     sprimal = if EnzymeCore.EnzymeRules.needs_primal(config) || EnzymeCore.EnzymeRules.needs_shadow(config)
#         func.val(a.val)
#     else
#         nothing
#     end
# 
#     shadow = if EnzymeCore.EnzymeRules.needs_shadow(config)
#         if EnzymeCore.EnzymeRules.width(config) == 1
#             (typeof(a) <: Const) ? EnzymeCore.make_zero(sprimal)::RT : func.val(a.dval)
#         else
#             ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
#                 Base.@_inline_meta
#                 (typeof(a) <: Const) ? EnzymeCore.make_zero(sprimal)::RT : func.val(a.dval[i])
#             end
#         end
#     else
#         nothing
#     end
# 
#     primal = if EnzymeCore.EnzymeRules.needs_primal(config)
#         sprimal
#     else
#         nothing
#     end
# 
#     P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
#     B = EnzymeCore.EnzymeRules.needs_primal(config) ? batch(Val(EnzymeCore.EnzymeRules.width(config)), RT) : Nothing
# 
#     return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, Nothing}(primal, shadow, nothing)
# end
# 
# function EnzymeCore.EnzymeRules.reverse(config,
#                                         func::EnzymeCore.Const{typeof(Oceananigans.Models.flattened_unique_values)},
#                                          ::Type{<:EnzymeCore.Annotation{RT}},
#                                          tape,
#                                          a) where RT
# 
#   return (nothing,)
# end

#####
##### launch!
#####

# function EnzymeCore.EnzymeRules.augmented_primal(config,
#                                                  func::EnzymeCore.Const{typeof(Oceananigans.Utils.launch!)},
#                                                  ::Type{EnzymeCore.Const{Nothing}},
#                                                  arch,
#                                                  grid,
#                                                  workspec,
#                                                  kernel!,
#                                                  kernel_args::Vararg{Any,N};
#                                                  include_right_boundaries = false,
#                                                  reduced_dimensions = (),
#                                                  location = nothing,
#                                                  active_cells_map = nothing,
#                                                  kwargs...) where N
# 
# 
#     workgroup, worksize = Oceananigans.Utils.work_layout(grid.val, workspec.val;
#                                                          include_right_boundaries,
#                                                          reduced_dimensions,
#                                                          location)
# 
#     offset = Oceananigans.Utils.offsets(workspec.val)
# 
#     if !isnothing(active_cells_map) 
#         workgroup, worksize = Oceananigans.Utils.active_cells_work_layout(workgroup, worksize, active_cells_map, grid.val) 
#         offset = nothing
#     end
# 
#     if worksize != 0
#       
#       # We can only launch offset kernels with Static sizes!!!!
# 
#       if isnothing(offset)
#           loop! = kernel!.val(Oceananigans.Architectures.device(arch.val), workgroup, worksize)
#           dloop! = (typeof(kernel!) <: EnzymeCore.Const) ? nothing : kernel!.dval(Oceananigans.Architectures.device(arch.val), workgroup, worksize)
#       else
#           loop! = kernel!.val(Oceananigans.Architectures.device(arch.val), KernelAbstractions.StaticSize(workgroup), Oceananigans.Utils.OffsetStaticSize(contiguousrange(worksize, offset))) 
#           dloop! = (typeof(kernel!) <: EnzymeCore.Const) ? nothing : kernel!.val(Oceananigans.Architectures.device(arch.val), KernelAbstractions.StaticSize(workgroup), Oceananigans.Utils.OffsetStaticSize(contiguousrange(worksize, offset)))
#       end
# 
#       @debug "Launching kernel $kernel! with worksize $worksize and offsets $offset from $workspec.val"
# 
# 
#       duploop = (typeof(kernel!) <: EnzymeCore.Const) ? EnzymeCore.Const(loop!) : EnzymeCore.Duplicated(loop!, dloop!)
# 
#       config2 = EnzymeCore.EnzymeRules.Config{#=needsprimal=#false, #=needsshadow=#false, #=width=#EnzymeCore.EnzymeRules.width(config), EnzymeCore.EnzymeRules.overwritten(config)[5:end]}()
#       subtape = EnzymeCore.EnzymeRules.augmented_primal(config2, duploop, EnzymeCore.Const{Nothing}, kernel_args...).tape
# 
#       tape = (duploop, subtape)
#     else
#       tape = nothing
#     end
# 
#     return EnzymeCore.EnzymeRules.AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, tape)
# end
# 
# @inline arg_elem_type(::Type{T}, ::Val{i}) where {T<:Tuple, i} = eltype(T.parameters[i])
# 
# function EnzymeCore.EnzymeRules.reverse(config::EnzymeCore.EnzymeRules.ConfigWidth{1},
#                                                 func::EnzymeCore.Const{typeof(Oceananigans.Utils.launch!)},
#                                                  ::Type{EnzymeCore.Const{Nothing}},
#                                                  tape,
#                                                  arch,
#                                                  grid,
#                                                  workspec,
#                                                  kernel!,
#                                                  kernel_args::Vararg{Any,N};
#                                                  include_right_boundaries = false,
#                                                  reduced_dimensions = (),
#                                                  location = nothing,
#                                                  active_cells_map = nothing,
#                                                  kwargs...) where N
# 
#   subrets = if tape !== nothing
#     duploop, subtape = tape
#     config2 = EnzymeCore.EnzymeRules.Config{#=needsprimal=#false, #=needsshadow=#false, #=width=#EnzymeCore.EnzymeRules.width(config), EnzymeCore.EnzymeRules.overwritten(config)[5:end]}()
#     EnzymeCore.EnzymeRules.reverse(config2, duploop, EnzymeCore.Const{Nothing}, subtape, kernel_args...)
#   else
#     res2 = ntuple(Val(N)) do i
#       Base.@_inline_meta
#       if kernel_args[i] isa Active
#         EnzymeCore.make_zero(kernel_args[i].val)
#       else
#         nothing
#       end
#     end
#   end
# 
#   subrets2 =  ntuple(Val(N)) do i
#       Base.@_inline_meta
#       if kernel_args[i] isa Active
#         subrets[i]::arg_elem_type(typeof(kernel_args), Val(i))
#       else
#         nothing
#       end
#     end
# 
#   return (nothing, nothing, nothing, nothing, subrets2...)
# 
# end

#####
##### update_model_field_time_series!
#####

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{typeof(Oceananigans.Models.update_model_field_time_series!)},
                                                 ::Type{EnzymeCore.Const{Nothing}},
                                                 model,
                                                 clock)

    time = (typeof(clock) <: Const) ? Const(Oceananigans.Utils.Time(clock.val.time)) : Duplicated(Oceananigans.Utils.Time(clock.val.time), Oceananigans.Utils.Time(clock.dval.time))

    possible_fts = Oceananigans.Models.possible_field_time_series(model.val)

    time_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(possible_fts)
    time_series_tuple = Oceananigans.Models.flattened_unique_values(time_series_tuple)

    fulltape = if EnzymeCore.EnzymeRules.width(config) == 1
        dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval)
        dtime_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(possible_fts)
        dtime_series_tuple = Oceananigans.Models.flattened_unique_values(dtime_series_tuple)

        tapes = []
        for (fts, dfts) in zip(time_series_tuple, dtime_series_tuple)
            dupft = Enzyme.Compiler.guaranteed_const(typeof(fts)) ? Const(fts) : Duplicated(fts, dfts)
            fwdfn, revfn = Enzyme.autodiff_thunk(EnzymeCore.ReverseSplitNoPrimal, Const{typeof(Oceananigans.Models.update_field_time_series!)}, Const, typeof(dupft), typeof(time))
            tape, primal, shadow = fwdfn(Const(Oceananigans.Models.update_field_time_series!), dupft, time)
            push!(tapes, tape)
        end
        tapes
    else
        ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
            Base.@_inline_meta
            dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval[i])
            dtime_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(possible_fts)
            dtime_series_tuple = Oceananigans.Models.flattened_unique_values(dtime_series_tuple)

            tapes = []
            for (fts, dfts) in zip(time_series_tuple, dtime_series_tuple)
                dupft = Enzyme.Compiler.guaranteed_const(typeof(fts)) ? Const(fts) : Duplicated(fts, dfts)
                fwdfn, revfn = Enzyme.autodiff_thunk(EnzymeCore.ReverseSplitNoPrimal, Const{typeof(Oceananigans.Models.update_field_time_series!)}, Const, typeof(dupft), typeof(time))
                tape, primal, shadow = fwdfn(Const(Oceananigans.Models.update_field_time_series!), dupft, time)
                push!(tapes, tape)
            end
            tapes
        end
    end

    return EnzymeCore.EnzymeRules.AugmentedReturn{Nothing, Nothing, Any}(nothing, nothing, fulltape::Any)
end

function EnzymeCore.EnzymeRules.reverse(config,
                                        func::EnzymeCore.Const{typeof(Oceananigans.Models.update_model_field_time_series!)},
                                        ::Type{EnzymeCore.Const{Nothing}},
                                        fulltape,
                                        model,
                                        clock)

    time = (typeof(clock) <: EnzymeCore.Const) ? Const(Oceananigans.Utils.Time(clock.val.time)) : Duplicated(Oceananigans.Utils.Time(clock.val.time), Oceananigans.Utils.Time(clock.dval.time))

    if EnzymeCore.EnzymeRules.width(config) == 1
        dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval)
        dtime_series_tuple = Oceananigans.OutputReaders.extract_field_timeseries(possible_fts)
        dtime_series_tuple = Oceananigans.Models.flattened_unique_values(dtime_series_tuple)

        tapes = fulltape
        i = 1
        for (fts, dfts) in zip(time_series_tuple, dtime_series_tuple)
            dupft = Enzyme.Compiler.guaranteed_const(typeof(fts)) ? Const(fts) : Duplicated(fts, dfts)
            fwdfn, revfn = Enzyme.autodiff_thunk(EnzymeCore.ReverseSplitNoPrimal, Const{typeof(Oceananigans.Models.update_field_time_series!)}, Const, typeof(dupft), typeof(time))
            revfn(Const(Oceananigans.Models.update_field_time_series!), dupft, time, tapes[i])
            push!(tapes, tape)
            i+= 1
        end
    else
        ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
            Base.@_inline_meta

            tapes = fulltape[i]
            dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval[i])
            dtime_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(possible_fts)
            dtime_series_tuple = Oceananigans.Models.flattened_unique_values(dtime_series_tuple)

            i += 1
            for (fts, dfts) in zip(time_series_tuple, dtime_series_tuple)
                dupft = Enzyme.Compiler.guaranteed_const(typeof(fts)) ? Const(fts) : Duplicated(fts, dfts)
                fwdfn, revfn = Enzyme.autodiff_thunk(EnzymeCore.ReverseSplitNoPrimal, Const{typeof(Oceananigans.Models.update_field_time_series!)}, Const, typeof(dupft), typeof(time))
                revfn(Const(Oceananigans.Models.update_field_time_series!), dupft, time, tapes[i])
                i+=1
            end
        end
    end

  return (nothing, nothing)

end

#####
##### top_tracer_boundary_conditions
#####

function EnzymeCore.EnzymeRules.augmented_primal(config,
                                                 func::EnzymeCore.Const{typeof(Oceananigans.Models.HydrostaticFreeSurfaceModels.top_tracer_boundary_conditions)},
                                                 ::RT, grid,
                                                 tracers) where RT

    primal = if EnzymeCore.EnzymeRules.needs_primal(config)
        func.val(grid.val, tracers.val)
    else
        nothing
    end

    shadow = if EnzymeCore.EnzymeRules.width(config) == 1
        func.val(grid.val, tracers.dval)
    else
        ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
            Base.@_inline_meta
            func.val(grid.val, tracers.dval)
        end
    end

    P = EnzymeCore.EnzymeRules.needs_primal(config) ? RT : Nothing
    B = batch(Val(EnzymeCore.EnzymeRules.width(config)), RT)
    return EnzymeCore.EnzymeRules.AugmentedReturn{P, B, Nothing}(primal, shadow, nothing)
end

function EnzymeCore.EnzymeRules.reverse(config,
                                        func::EnzymeCore.Const{typeof(Oceananigans.Models.HydrostaticFreeSurfaceModels.top_tracer_boundary_conditions)},
                                        ::RT, grid,
                                        tracers) where RT
    return (nothing, nothing)
end

end
