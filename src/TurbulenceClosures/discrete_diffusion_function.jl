using Oceananigans.Operators: interpolate
using Oceananigans.Utils: instantiate

# To allow indexing a diffusivity with (i, j, k, grid, Lx, Ly, Lz)
struct DiscreteDiffusionFunction{LX, LY, LZ, P, F} 
    func :: F
    parameters :: P

    function DiscreteDiffusionFunction{LX, LY, LZ}(func::F, parameters::P) where {LX, LY, LZ, F, P}
        return new{LX, LY, LZ, P, F}(func, parameters)
    end
end

function DiscreteDiffusionFunction(func; parameters, loc)
    loc = instantiate.(loc)
    return DiscreteDiffusionFunction{typeof(loc[1]), typeof(loc[2]), typeof(loc[3])}(func, parameters)
end

const UnparameterizedDDF           = DiscreteDiffusionFunction{<:Any, <:Any, <:Any,  <:Nothing}
const UnlocalizedDDF               = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing}
const UnlocalizedUnparametrizedDDF = DiscreteDiffusionFunction{<:Nothing, <:Nothing, <:Nothing, <:Nothing}

@inline function getdiffusion(dd::DiscreteDiffusionFunction{LX, LY, LZ}, 
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ} 
        from = (LX(), LY(), LZ())
        return interpolate(i, j, k, grid, from, location, dd.func, clock, fields, dd.parameters)
end

@inline function getdiffusion(dd::UnparameterizedDDF{LX, LY, LZ}, 
                              i, j, k, grid, location, clock, fields) where {LX, LY, LZ} 
        from = (LX(), LY(), LZ())
    return interpolate(i, j, k, grid, from, location, dd.func, clock, fields)
end

@inline getdiffusion(dd::UnlocalizedDDF, i, j, k, grid, location, clock, fields) = 
        dd.func(i, j, k, grid, location..., clock, fields, dd.parametrs)

@inline getdiffusion(dd::UnlocalizedUnparametrizedDDF, i, j, k, grid, location, clock, fields) = 
        dd.func(i, j, k, grid, location..., clock, fields)

Adapt.adapt_structure(to, dd::DiscreteDiffusionFunction{LX, LY, LZ}) where {LX, LY, LZ} =
     DiscreteBoundaryFunction{LX, LY, LZ}(Adapt.adapt(to, dd.func),
                                          Adapt.adapt(to, dd.parameters))
