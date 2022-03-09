# using CUDA: synchronize

# multiregion_transformations = Dict{Symbol, Symbol}()

# multiregion_transformations[:AbstractGrid]          = :MultiRegionGrid
# multiregion_transformations[:RectilinearGrid]       = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:RectilinearGrid}")
# multiregion_transformations[:LatitudeLongitudeGrid] = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}")
# multiregion_transformations[:ImmersedBoundaryGrid]  = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}")

# multiregion_transformations[:AbstractField] = :MultiRegionField
# multiregion_transformations[:Field]         = :MultiRegionField

# # For non-returning functions -> can we make it NON BLOCKING? This seems to be synchronous!
# function apply_regionally!(func!, args...; kwargs...)
#     mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
#     mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
#     isnothing(mra) && isnothing(mrk) && return func!(args...; kwargs...)

#     if isnothing(mra) 
#         devs = devices(mrk)
#     else
#         devs = devices(mra)
#     end
    
#     @sync for (r, dev) in enumerate(devs)
#         # @sync begin
#         # r = 1
#         # dev = devs[1]
#         @async begin
#             switch_device!(dev)
#             region_args = Tuple(getregion(arg, r) for arg in args)
#             region_kwargs = Tuple(getregion(kwarg, r) for kwarg in kwargs)
#             func!(region_args...; region_kwargs...)
#         end
#     end
# end 

# # For functions with return statements -> BLOCKING! (use as seldom as possible)
# function construct_regionally(constructor, args...; kwargs...)
#     mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
#     mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
#     isnothing(mra) && isnothing(mrk) && return func(args...; kwargs...)

#     if isnothing(mra) 
#         devs = devices(mrk)
#     else
#         devs = devices(mra)
#     end

#     res = Tuple((switch_device!(dev);
#                 region_args = Tuple(getregion(arg, r) for arg in args);
#                 region_kwargs = Tuple(getregion(kwarg, r) for kwarg in kwargs);
#                 constructor(region_args...; region_kwargs...))
#                 for (r, dev) in enumerate(devs))

#     sync_all_devices!(devs)

#     return MultiRegionObject(Tuple(res), devs)
# end

# function sync_all_devices!(devices)
#     @sync for dev in devices
#         @async begin
#             switch_device!(dev)
#             sync_device!(dev)
#         end
#     end
# end

# sync_device!(::CuDevice) = synchronize(blocking=false)
# sync_device!(::CPU)      = nothing

# redispatch(arg::Symbol) = arg

# function redispatch(arg::Expr)
#     arg.head === :(::) || return arg # not a type definition? (this shouldn't be possible)
#     argtype = arg.args[2]
#     # If argtype is defined in multiregion_transformation, replace it.
#     # Otherwise, keep the original.
#     if argtype ∈ keys(multiregion_transformations)
#         newargtype = multiregion_transformations[argtype]
#         arg.args[2] = newargtype
#     end
#     return arg
# end

# # @regional is not used at the moment
# # Before it is better to validate with just applying apply_regionally!

# macro regional(expr)
#     expr.head ∈ (:(=), :function) || error("@regional can only prefix function definitions")
    
#     # Consistently to @kernel, also @regional does not allow returniong statements
#     last_expr = expr.args[end].args[end]
#     if last_expr.head == :return && !(last_expr.args[1] ∈ (nothing,  :nothing))
#         error("Return statement not permitted in a regional function!")
#     end

#     original_expr  = deepcopy(expr)
#     oldargs        = original_expr.args[1].args[2:end]
   
#     # Now redefine expr into its multiregion version
#     multiregion_expr = expr 
#     fdef  = multiregion_expr.args[1]
#     fname = fdef.args[1]        
#     fargs = fdef.args[2:end]
#     newargs = [redispatch(arg) for arg in fargs]

#     # If there is no dispatch do not implemement the multi_region version 
#     different_multiregion_version = newargs != oldargs
    
#     if different_multiregion_version
#         multiregion_expr = quote
#             $(fdef) = apply_regionally!($(fname), $(newargs...))
#         end
#         return quote
#             $(esc(original_expr))
#             $(esc(multiregion_expr))
#         end
#     else
#         return quote 
#             $(esc(original_expr))
#         end
#     end
# end

# macro apply_regionally(expr)
#     if expr.head == :call
#         func = expr.args[1]
#         args = expr.args[2:end]
#         multi_region = quote
#             apply_regionally!($func, $(args...))
#         end
#         return quote
#             $(esc(multi_region))
#         end
#     elseif expr.head == :block
#         new_expr = deepcopy(expr)
#         for (idx, arg) in enumerate(expr.args)
#             if arg isa Expr && arg.head == :call
#                 func = arg.args[1]
#                 args = arg.args[2:end]
#                 new_expr.args[idx] = quote
#                     apply_regionally!($func, $(args...))
#                 end
#             end
#         end
#         return quote
#             $(esc(new_expr))
#         end
#     end
# end