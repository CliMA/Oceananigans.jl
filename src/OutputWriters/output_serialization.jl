#####
##### Safe materialization of serialized output metadata
#####

# NetCDF and Zarr attributes can only store a limited set of Julia values directly.
# Grid reconstruction therefore serializes types, topologies, and a few constructors as
# strings. Decode only the syntax emitted by the writers: output files are data, not code.

const serialized_output_values = Dict{Symbol, Any}(
    :nothing => nothing,
    :Nothing => Nothing,
    :missing => missing,
    Symbol("true") => true,
    Symbol("false") => false,
    :Inf => Inf,
    :NaN => NaN,
    :CPU => CPU,
    :GPU => GPU,
    :Center => Center,
    :Face => Face,
    :Flat => Flat,
    :Periodic => Periodic,
    :Bounded => Bounded,
    :FullyConnected => FullyConnected,
    :LeftConnected => LeftConnected,
    :RightConnected => RightConnected,
    :RightFaceFolded => RightFaceFolded,
    :RightCenterFolded => RightCenterFolded,
    :LeftConnectedRightCenterFolded => LeftConnectedRightCenterFolded,
    :LeftConnectedRightFaceFolded => LeftConnectedRightFaceFolded,
    :LeftConnectedRightCenterConnected => LeftConnectedRightCenterConnected,
    :LeftConnectedRightFaceConnected => LeftConnectedRightFaceConnected,
    :RectilinearGrid => RectilinearGrid,
    :LatitudeLongitudeGrid => LatitudeLongitudeGrid,
    :OrthogonalSphericalShellGrid => OrthogonalSphericalShellGrid,
    :TripolarGrid => TripolarGrid,
    :RotatedLatitudeLongitudeGrid => RotatedLatitudeLongitudeGrid,
    :ConformalCubedSpherePanelGrid => ConformalCubedSpherePanelGrid,
    :LambertConformalConicGrid => LambertConformalConicGrid,
    :GridFittedBoundary => GridFittedBoundary,
    :GridFittedBottom => GridFittedBottom,
    :PartialCellBottom => PartialCellBottom,
    :CenterImmersedCondition => CenterImmersedCondition,
    :InterfaceImmersedCondition => InterfaceImmersedCondition,
    :Tripolar => Tripolar,
    :LatitudeLongitudeRotation => LatitudeLongitudeRotation,
    :CubedSphereConformalMapping => CubedSphereConformalMapping,
    :LambertConformalConic => LambertConformalConic,
    :ColumnEnsembleSize => ColumnEnsembleSize,
    :Colon => Colon,
    :Tuple => Tuple,
    :Bool => Bool,
    :Float16 => Float16,
    :Float32 => Float32,
    :Float64 => Float64,
    :Int8 => Int8,
    :Int16 => Int16,
    :Int32 => Int32,
    :Int64 => Int64,
    :Int128 => Int128,
    :UInt8 => UInt8,
    :UInt16 => UInt16,
    :UInt32 => UInt32,
    :UInt64 => UInt64,
    :UInt128 => UInt128,
)

const serialized_output_namespaces = Set((
    :Oceananigans,
    :Architectures,
    :Grids,
    :ImmersedBoundaries,
    :OrthogonalSphericalShellGrids,
))

function materialize_serialized_output(string::AbstractString)
    expression = Meta.parse(string)
    return materialize_serialized_output(expression)
end

materialize_serialized_output(value::Union{Number, Bool}) = value
materialize_serialized_output(value::QuoteNode) = materialize_serialized_output(value.value)

function materialize_serialized_output(symbol::Symbol)
    haskey(serialized_output_values, symbol) ||
        throw(ArgumentError("Unsupported serialized output value: $symbol"))
    return serialized_output_values[symbol]
end

function materialize_serialized_output(expression::Expr)
    head = expression.head

    if head === :tuple
        return tuple(map(materialize_serialized_output, expression.args)...)
    elseif head === :vect
        return [map(materialize_serialized_output, expression.args)...]
    elseif head === :ref
        return materialize_typed_output_vector(expression)
    elseif head === :.
        return materialize_qualified_output_value(expression)
    elseif head === :curly
        return materialize_parameterized_output_type(expression)
    elseif head === :call
        return materialize_serialized_output_call(expression)
    end

    throw(ArgumentError("Unsupported serialized output syntax: $head"))
end

function materialize_typed_output_vector(expression)
    element_type = materialize_serialized_output(first(expression.args))
    element_type <: Number ||
        throw(ArgumentError("Unsupported serialized output array element type: $element_type"))
    values = map(materialize_serialized_output, expression.args[2:end])
    return element_type[values...]
end

function materialize_qualified_output_value(expression)
    names = qualified_output_names(expression)
    all(name ∈ serialized_output_namespaces for name in names[1:end-1]) ||
        throw(ArgumentError("Unsupported serialized output namespace: $(join(names[1:end-1], '.'))"))
    return materialize_serialized_output(last(names))
end

qualified_output_names(symbol::Symbol) = (symbol,)
qualified_output_names(node::QuoteNode) = qualified_output_names(node.value)

function qualified_output_names(expression::Expr)
    expression.head === :. ||
        throw(ArgumentError("Unsupported qualified serialized output value"))
    left = qualified_output_names(expression.args[1])
    right = qualified_output_names(expression.args[2])
    return (left..., right...)
end

function materialize_parameterized_output_type(expression)
    base_type = materialize_serialized_output(first(expression.args))
    parameters = map(materialize_serialized_output, expression.args[2:end])

    if base_type === Tuple
        return Tuple{parameters...}
    elseif base_type === ColumnEnsembleSize
        return ColumnEnsembleSize{parameters...}
    end

    throw(ArgumentError("Unsupported serialized parametric type: $base_type"))
end

function materialize_serialized_output_call(expression)
    callable_expression = first(expression.args)
    arguments = map(materialize_serialized_output, expression.args[2:end])

    if callable_expression === :(:)
        return materialize_serialized_output_range(arguments)
    elseif callable_expression === :- || callable_expression === :+
        length(arguments) == 1 ||
            throw(ArgumentError("A serialized numeric sign must have one argument"))
        value = only(arguments)
        value isa Number ||
            throw(ArgumentError("A serialized numeric sign must apply to a number"))
        return callable_expression === :- ? -value : +value
    end

    callable = materialize_serialized_output(callable_expression)
    serialized_output_callable(callable) ||
        throw(ArgumentError("Unsupported serialized output constructor: $callable"))
    return callable(arguments...)
end

serialized_output_callable(callable) = callable === CPU ||
                                       callable === GPU ||
                                       callable === Colon ||
                                       callable === CenterImmersedCondition ||
                                       callable === InterfaceImmersedCondition ||
                                       (callable isa Type && callable <: ColumnEnsembleSize)

function materialize_serialized_output_range(arguments)
    length(arguments) == 2 && return arguments[1]:arguments[2]
    length(arguments) == 3 && return arguments[1]:arguments[2]:arguments[3]
    throw(ArgumentError("A serialized range must contain two or three values"))
end
