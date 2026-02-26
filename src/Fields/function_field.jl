struct FunctionField{LX, LY, LZ, C, P, F, G, T, D} <: AbstractField{LX, LY, LZ, G, T, 3}
          func :: F
          grid :: G
         clock :: C
    parameters :: P

    @doc """
        FunctionField{LX, LY, LZ}(func, grid; clock=nothing, parameters=nothing, discrete_form=false) where {LX, LY, LZ}

    Returns a `FunctionField` on `grid` and at location `LX, LY, LZ`.

    If `clock` is not specified, then `func` must be a function with signature
    `func(x, y, z)`. If clock is specified, `func` must be a function with signature
    `func(x, y, z, t)`, where `t` is internally determined from `clock.time`.

    A `FunctionField` will return the result of `func(x, y, z [, t])` at `LX, LY, LZ` on
    `grid` when indexed at `i, j, k`.

    If `discrete_form=true`, then `func` must be a function with signature `func(i, j, k, grid)`,
    `func(i, j, k, grid, clock)`, or `func(i, j, k, grid, clock, parameters)` depending on whether
    `clock` or `parameters` are specified.
    """
    @inline function FunctionField{LX, LY, LZ}(func::F,
                                               grid::G;
                                               clock::C=nothing,
                                               parameters::P=nothing,
                                               discrete_form::Bool=false) where {LX, LY, LZ, F, G, C, P}
        FT = eltype(grid)
        return new{LX, LY, LZ, C, P, F, G, FT, discrete_form}(func, grid, clock, parameters)
    end

    @inline function FunctionField{LX, LY, LZ}(f::FunctionField,
                                               grid::G;
                                               clock::C=nothing,
                                               discrete_form::Bool=false) where {LX, LY, LZ, G, C}
        P = typeof(f.parameters)
        T = eltype(grid)
        F = typeof(f.func)
        return new{LX, LY, LZ, C, P, F, G, T, discrete_form}(f.func, grid, clock, f.parameters)
    end
end

const DiscreteFunctionField{LX, LY, LZ, C, P, F, G, T} = FunctionField{LX, LY, LZ, C, P, F, G, T, true}

Adapt.parent_type(T::Type{<:FunctionField}) = T

"""Return `a`, or convert `a` to `FunctionField` if `a::Function`"""
fieldify_function(L, a, grid) = a
fieldify_function(L, a::Function, grid) = FunctionField(L, a, grid)

# This is a convenience form with `L` as positional argument.
@inline FunctionField(L::Tuple{<:Type, <:Type, <:Type}, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)
@inline FunctionField(L::Tuple{LX, LY, LZ}, func, grid) where {LX, LY, LZ}= FunctionField{LX, LY, LZ}(func, grid)

@inline indices(::FunctionField) = (:, :, :)

@inline has_discrete_form(f::FunctionField) = has_discrete_form(typeof(f))
@inline has_discrete_form(::Type{<:FunctionField}) = false
@inline has_discrete_form(::Type{<:DiscreteFunctionField}) = true

# Various possibilities for calling FunctionField.func:
@inline call_func(clock,     parameters, func, x...) = func(x..., clock.time, parameters)
@inline call_func(clock,     ::Nothing,  func, x...) = func(x..., clock.time)
@inline call_func(::Nothing, parameters, func, x...) = func(x..., parameters)
@inline call_func(::Nothing, ::Nothing,  func, x...) = func(x...)
# If clock is specified for discrete form, pass directly to `func`
@inline call_func(clock,     parameters, func, i, j, k, grid::AbstractGrid) = func(i, j, k, grid, clock, parameters)
@inline call_func(clock,     ::Nothing,  func, i, j, k, grid::AbstractGrid) = func(i, j, k, grid, clock)

@inline function Base.getindex(f::FunctionField{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    f_ijk = call_func(f.clock, f.parameters, f.func, node(i, j, k, f.grid, LX(), LY(), LZ())...)
    return convert(eltype(f.grid), f_ijk)
end

@inline function Base.getindex(f::DiscreteFunctionField{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    f_ijk = call_func(f.clock, f.parameters, f.func, i, j, k, f.grid)
    return convert(eltype(f.grid), f_ijk)
end

@inline (f::FunctionField)(x...) = call_func(f.clock, f.parameters, f.func, x...)

Adapt.adapt_structure(to, f::FunctionField{LX, LY, LZ}) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(Adapt.adapt(to, f.func),
                           Adapt.adapt(to, f.grid),
                           clock = Adapt.adapt(to, f.clock),
                           parameters = Adapt.adapt(to, f.parameters),
                           discrete_form = has_discrete_form(f))


Architectures.on_architecture(to, f::FunctionField{LX, LY, LZ}) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(on_architecture(to, f.func),
                              on_architecture(to, f.grid),
                              clock = on_architecture(to, f.clock),
                              parameters = on_architecture(to, f.parameters),
                              discrete_form = has_discrete_form(f))

Base.show(io::IO, field::FunctionField) =
    print(io, "FunctionField located at ", show_location(field), "\n",
          "├── func: $(prettysummary(field.func))", "\n",
          "├── grid: $(summary(field.grid))\n",
          "├── clock: $(summary(field.clock))\n",
          "└── parameters: $(field.parameters)")
