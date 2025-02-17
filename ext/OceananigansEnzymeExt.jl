module OceananigansEnzymeExt

using Oceananigans
using Oceananigans.Fields: FunctionField
using Oceananigans.Utils: contiguousrange
#Oceananigans.Models.HydrostaticFreeSurfaceModels.top_tracer_boundary_conditions

using KernelAbstractions

isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

using Enzyme: EnzymeCore
using Enzyme.EnzymeCore: Active, Const, Duplicated

EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Base.:(==)), ::Oceananigans.AbstractGrid, ::Oceananigans.AbstractGrid) = nothing
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.AbstractOperations.validate_grid), x...) = nothing
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.AbstractOperations.metric_function), x...) = nothing
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.flatten_reduced_dimensions), x...) = nothing
EnzymeCore.EnzymeRules.inactive_noinl(::typeof(Oceananigans.Utils.prettytime), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.Grids.total_size), x...) = nothing
EnzymeCore.EnzymeRules.inactive(::typeof(Oceananigans.BoundaryConditions.parent_size_and_offset), x...) = nothing
@inline EnzymeCore.EnzymeRules.inactive_type(v::Type{Oceananigans.Utils.KernelParameters}) = true

@inline batch(::Val{1}, ::Type{T}) where T = T
@inline batch(::Val{N}, ::Type{T}) where {T, N} = NTuple{N, T}

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
    
    possible_fts = Oceananigans.Models.possible_field_time_series(model.val)

    time_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(possible_fts)
    time_series_tuple = Oceananigans.Models.flattened_unique_values(time_series_tuple)

    if EnzymeCore.EnzymeRules.width(config) == 1
        dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval)
        dtime_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(dpossible_fts)
        dtime_series_tuple = Oceananigans.Models.flattened_unique_values(dtime_series_tuple)

        tapes = fulltape
        i = 1
        for (fts, dfts) in zip(time_series_tuple, dtime_series_tuple)
            dupft = Enzyme.Compiler.guaranteed_const(typeof(fts)) ? Const(fts) : Duplicated(fts, dfts)
            fwdfn, revfn = Enzyme.autodiff_thunk(EnzymeCore.ReverseSplitNoPrimal, Const{typeof(Oceananigans.Models.update_field_time_series!)}, Const, typeof(dupft), typeof(time))
            revfn(Const(Oceananigans.Models.update_field_time_series!), dupft, time, tapes[i])
            i+= 1
        end
    else
        ntuple(Val(EnzymeCore.EnzymeRules.width(config))) do i
            Base.@_inline_meta

            tapes = fulltape[i]
            dpossible_fts = Oceananigans.Models.possible_field_time_series(model.dval[i])
            dtime_series_tuple = Oceananigans.OutputReaders.extract_field_time_series(dpossible_fts)
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
