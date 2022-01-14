struct FunctionField{LX, LY, LZ, C, P, F, G, T} <: AbstractField{LX, LY, LZ, G, T, 3}
          func :: F
          grid :: G
         clock :: C
    parameters :: P

    @doc """
        FunctionField{LX, LY, LZ}(func, grid; clock=nothing, parameters=nothing) where {LX, LY, LZ}

    Returns a `FunctionField` on `grid` and at location `LX, LY, LZ`.

    If `clock` is not specified, then `func` must be a function with signature
    `func(x, y, z)`. If clock is specified, `func` must be a function with signature
    `func(x, y, z, t)`, where `t` is internally determined from `clock.time`.

    A `FunctionField` will return the result of `func(x, y, z [, t])` at `LX, LY, LZ` on
    `grid` when indexed at `i, j, k`.
    """
    function FunctionField{LX, LY, LZ}(func::F, grid::G; clock::C=nothing, parameters::P=nothing) where {LX, LY, LZ, F, G, C, P}
        T = eltype(grid)
        return new{LX, LY, LZ, C, P, F, G, T}(func, grid, clock, parameters)
    end

    @doc """
        FunctionField{LX, LY, LZ}(func::FunctionField, grid; clock) where {LX, LY, LZ}

    Adds `clock` to an existing `FunctionField` and relocates it to `(LX, LY, LZ)` on `grid`.
    """
    function FunctionField{LX, LY, LZ}(f::FunctionField, grid::G; clock::C=nothing) where {LX, LY, LZ, G, C}
        P = typeof(f.parameters)
        T = eltype(grid)
        F = typeof(f.func)
        return new{LX, LY, LZ, C, P, F, G, T}(f.func, grid, clock, f.parameters)
    end
end

"""Return `a`, or convert `a` to `FunctionField` if `a::Function`"""
fieldify_function(L, a, grid) = a
fieldify_function(L, a::Function, grid) = FunctionField(L, a, grid)

"""
    FunctionField(L::Tuple, func, grid)

Returns a stationary `FunctionField` on `grid` and at location `L = (LX, LY, LZ)`,
where `func` is callable with signature `func(x, y, z)`.
"""
FunctionField(L::Tuple, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)

# Various possibilities for calling FunctionField.func:
@inline call_func(clock, parameters, func, x, y, z)     = func(x, y, z, clock.time, parameters)
@inline call_func(::Nothing, parameters, func, x, y, z) = func(x, y, z, parameters)
@inline call_func(clock, ::Nothing, func, x, y, z)      = func(x, y, z, clock.time)
@inline call_func(::Nothing, ::Nothing, func, x, y, z)  = func(x, y, z)

# For setting ReducedField
@inline call_func(::Nothing, ::Nothing, func, x, y)     = func(x, y)
@inline call_func(::Nothing, ::Nothing, func, x)        = func(x)

@inline Base.getindex(f::FunctionField{LX, LY, LZ}, i, j, k) where {LX, LY, LZ} =
    call_func(f.clock, f.parameters, f.func, node(LX(), LY(), LZ(), i, j, k, f.grid)...)

@inline (f::FunctionField)(x...) = call_func(f.clock, f.parameters, f.func, x...)

Adapt.adapt_structure(to, f::FunctionField{LX, LY, LZ}) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(Adapt.adapt(to, f.func),
                           Adapt.adapt(to, f.grid),
                           clock = Adapt.adapt(to, f.clock),
                           parameters = Adapt.adapt(to, f.parameters))

Base.show(io::IO, field::FunctionField) =
    print(io, "FunctionField located at ", show_location(field), '\n',
          "├── func: $(short_show(field.func))", '\n',
          "├── grid: $(short_show(field.grid))\n",
          "├── clock: $(short_show(field.clock))\n",
          "└── parameters: $(field.parameters)")

