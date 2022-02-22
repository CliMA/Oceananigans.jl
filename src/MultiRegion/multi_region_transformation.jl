multiregion_transformations = Dict{Symbol, Symbol}()

multiregion_transformations[:AbstractGrid]          = :MultiRegionGrid
multiregion_transformations[:RectilinearGrid]       = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:RectilinearGrid}")
multiregion_transformations[:LatitudeLongitudeGrid] = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:LatitudeLongitudeGrid}")
multiregion_transformations[:ImmersedBoundaryGrid]  = Symbol("MultiRegionGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}")

multiregion_transformations[:AbstractField] = :MultiRegionField
multiregion_transformations[:Field]         = :MultiRegionField

# For non-returning functions
function apply_regionally!(func!, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func(args...; kwargs...)

    @sync for (r, dev) in enumerate(devices(mra))
        @async begin
            switch_device!(dev);
            region_args = Tuple(getregion(arg, r) for arg in args);
            region_kwargs = NamedTuple{keys(kwargs)}(getregion(kwarg, r) for kwarg in kwargs);
            func!(region_args...; region_kwargs...)
        end
    end
    
end
 
# For functions with return statements
function apply_regionally(func, args...; kwargs...)
    mra = isnothing(findfirst(isregional, args)) ? nothing : args[findfirst(isregional, args)]
    mrk = isnothing(findfirst(isregional, kwargs)) ? nothing : kwargs[findfirst(isregional, kwargs)]
    isnothing(mra) && isnothing(mrk) && return func(args...; kwargs...)

    res = Tuple((switch_device!(dev);
                 region_args = Tuple(getregion(arg, r) for arg in args);
                 region_kwargs = NamedTuple{keys(kwargs)}(getregion(kwarg, r) for kwarg in kwargs);
                 func(region_args...; region_kwargs...))
                 for (r, dev) in enumerate(devices(mra)))
    
    return MultiRegionObject(res, devices(mra))
end

redispatch(arg::Symbol) = arg

function redispatch(arg::Expr)
    arg.head === :(::) || return arg # not a type definition? (this shouldn't be possible)
    argtype = arg.args[2]
    # If argtype is defined in multiregion_transformation, replace it.
    # Otherwise, keep the original.
    if argtype ∈ keys(multiregion_transformations)
        newargtype = multiregion_transformations[argtype]
        arg.args[2] = newargtype
    end
    return arg
end

# @regional is not used at the moment
macro regional(expr)
    expr.head ∈ (:(=), :function) || error("@regional can only prefix function definitions")
    
    original_expr  = deepcopy(expr)
    oldargs        = original_expr.args[1].args[2:end]
   
    # Now redefine expr into its multiregion version
    multiregion_expr = expr 
    fdef  = multiregion_expr.args[1]
    fname = fdef.args[1]        
    fargs = fdef.args[2:end]
    newargs = [redispatch(arg) for arg in fargs]

    # If there is no dispatch do not implemement the multi_region version 
    different_multiregion_version = newargs != oldargs
    
    if different_multiregion_version
        last(string(fname)) == '!' ? regionalize = :apply_regionally! : regionalize = :apply_regionally
        multiregion_expr = quote
            $(fdef) = $(regionalize)($(fname), $(newargs...))
        end
        return quote
            $(esc(original_expr))
            $(esc(multiregion_expr))
        end
    else
        return quote 
            $(esc(original_expr))
        end
    end
end