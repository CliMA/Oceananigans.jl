struct FunctionField{LX, LY, LZ, C, P, F, G, I, T} <: AbstractField{LX, LY, LZ, G, T, 3}
          func :: F
          grid :: G
         clock :: C
    parameters :: P
       indices :: I

    @doc """
        FunctionField{LX, LY, LZ}(func, grid; clock=nothing, parameters=nothing, indices=(:, :, :)) where {LX, LY, LZ}

    Returns a `FunctionField` on `grid` and at location `LX, LY, LZ`.

    If `clock` is not specified, then `func` must be a function with signature
    `func(x, y, z)`. If clock is specified, `func` must be a function with signature
    `func(x, y, z, t)`, where `t` is internally determined from `clock.time`.

    A `FunctionField` will return the result of `func(x, y, z [, t])` at `LX, LY, LZ` on
    `grid` when indexed at `i, j, k`.

    `indices` restricts the region of the field that is considered active (e.g.,
    `indices = (:, :, Nz+1)` marks the field as living on a single horizontal slice).
    The function signature is always `func(x, y, z [, t])` regardless of `indices`.

    Examples
    ========

    Default `indices` — the `indices` line is absent from the output:

    ```jldoctest
    julia> using Oceananigans

    julia> using Oceananigans.Fields: FunctionField

    julia> f(x, y, z) = sin(x) * cos(y) + z;

    julia> clock = Clock(time=0.0);

    julia> grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1));

    julia> FunctionField{Center, Center, Center}(f, grid; clock)
    FunctionField located at (Center, Center, Center)
    ├── func: f (generic function with 1 method)
    ├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
    ├── clock: Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)
    └── parameters: nothing
    ```

    Reduced `indices` — the active footprint is a single horizontal slice:

    ```jldoctest
    julia> using Oceananigans

    julia> using Oceananigans.Fields: FunctionField

    julia> η(x, y, z) = sin(x) * cos(y);

    julia> clock = Clock(time=0.0);

    julia> grid = RectilinearGrid(size=(1, 1, 1), extent=(1, 1, 1));

    julia> FunctionField{Center, Center, Face}(η, grid; clock, indices=(:, :, grid.Nz+1))
    FunctionField located at (Center, Center, Face)
    ├── func: η (generic function with 1 method)
    ├── grid: 1×1×1 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 1×1×1 halo
    ├── clock: Clock{Float64, Float64}(time=0 seconds, iteration=0, last_Δt=Inf days)
    ├── indices: (:, :, 2)
    └── parameters: nothing
    ```
    """
    @inline function FunctionField{LX, LY, LZ}(func::F,
                                               grid::G;
                                               clock::C=nothing,
                                               parameters::P=nothing,
                                               indices::I=(:, :, :)) where {LX, LY, LZ, F, G, C, P, I}
        FT = eltype(grid)
        return new{LX, LY, LZ, C, P, F, G, I, FT}(func, grid, clock, parameters, indices)
    end

    @inline function FunctionField{LX, LY, LZ}(f::FunctionField,
                                               grid::G;
                                               clock::C=nothing) where {LX, LY, LZ, G, C}
        P = typeof(f.parameters)
        T = eltype(grid)
        F = typeof(f.func)
        I = typeof(f.indices)
        return new{LX, LY, LZ, C, P, F, G, I, T}(f.func, grid, clock, f.parameters, f.indices)
    end
end

Adapt.parent_type(T::Type{<:FunctionField}) = T

"""Return `a`, or convert `a` to `FunctionField` if `a::Function`"""
fieldify_function(L, a, grid) = a
fieldify_function(L, a::Function, grid) = FunctionField(L, a, grid)

# This is a convenience form with `L` as positional argument.
@inline FunctionField(L::Tuple{<:Type, <:Type, <:Type}, func, grid) = FunctionField{L[1], L[2], L[3]}(func, grid)
@inline FunctionField(L::Tuple{LX, LY, LZ}, func, grid) where {LX, LY, LZ}= FunctionField{LX, LY, LZ}(func, grid)

@inline indices(f::FunctionField) = f.indices

# Various possibilities for calling FunctionField.func:
@inline call_func(clock,     parameters, func, x...) = func(x..., clock.time, parameters)
@inline call_func(clock,     ::Nothing,  func, x...) = func(x..., clock.time)
@inline call_func(::Nothing, parameters, func, x...) = func(x..., parameters)
@inline call_func(::Nothing, ::Nothing,  func, x...) = func(x...)

@inline function Base.getindex(f::FunctionField{LX, LY, LZ}, i, j, k) where {LX, LY, LZ}
    f_ijk = call_func(f.clock, f.parameters, f.func, node(i, j, k, f.grid, LX(), LY(), LZ())...)
    return convert(eltype(f.grid), f_ijk)
end

@inline (f::FunctionField)(x...) = call_func(f.clock, f.parameters, f.func, x...)

Adapt.adapt_structure(to, f::FunctionField{LX, LY, LZ}) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(Adapt.adapt(to, f.func),
                           Adapt.adapt(to, f.grid),
                           clock      = Adapt.adapt(to, f.clock),
                           parameters = Adapt.adapt(to, f.parameters),
                           indices    = f.indices)


Architectures.on_architecture(to, f::FunctionField{LX, LY, LZ}) where {LX, LY, LZ} =
    FunctionField{LX, LY, LZ}(on_architecture(to, f.func),
                              on_architecture(to, f.grid),
                              clock      = on_architecture(to, f.clock),
                              parameters = on_architecture(to, f.parameters),
                              indices    = f.indices)

function Base.show(io::IO, field::FunctionField)
    idx = indices_summary(field)
    idx_str = idx == "(:, :, :)" ? "" : "├── indices: $idx\n"
    print(io, "FunctionField located at ", show_location(field), "\n",
          "├── func: $(prettysummary(field.func))", "\n",
          "├── grid: $(summary(field.grid))\n",
          "├── clock: $(summary(field.clock))\n",
          idx_str,
          "└── parameters: $(field.parameters)")
end
