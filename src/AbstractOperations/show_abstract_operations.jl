import Oceananigans: short_show
using Oceananigans.Grids: domain_string
using Oceananigans.Fields: show_location

for op_string in ("UnaryOperation", "BinaryOperation", "MultiaryOperation", "Derivative", "KernelFunctionOperation")
    op = eval(Symbol(op_string))
    @eval begin
        operation_name(::$op) = $op_string
    end
end

operation_name(op::GridMetricOperation)  = string(op.metric)
operation_name(mo::ConditionalOperation) = "Conditioned Field"

function show_interp(op)
    op_str = string(op)
    if op_str[1:8] == "identity"
        return "identity"
    else
        return op_str
    end
end

short_show(operation::AbstractOperation) = string(operation_name(operation), " at ", show_location(operation))

Base.show(io::IO, operation::AbstractOperation) =
    print(io,
          short_show(operation), '\n',
          "├── grid: ", short_show(operation.grid), '\n',
          "│   └── domain: ", domain_string(operation.grid), '\n',
          "└── tree: ", "\n", "    ", tree_show(operation, 1, 0))

"Return a representation of number or function leaf within a tree visualization of an `AbstractOperation`."
tree_show(a::Union{Number, Function}, depth, nesting) = string(a)

"Fallback for displaying a leaf within a tree visualization of an `AbstractOperation`."
tree_show(a, depth, nesting) = short_show(a) # fallback

"Returns a string corresponding to padding characters for a tree visualization of an `AbstractOperation`."
get_tree_padding(depth, nesting) = "    "^(depth-nesting) * "│   "^nesting

"Return a string representaion of a `UnaryOperation` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(unary::UnaryOperation{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = get_tree_padding(depth, nesting)

    return string(unary.op, " at ", show_location(X, Y, Z), " via ", show_interp(unary.▶), '\n',
                  padding, "└── ", tree_show(unary.arg, depth+1, nesting))
end

"Return a string representaion of a `BinaryOperation` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(binary::BinaryOperation{X, Y, Z}, depth, nesting) where {X, Y, Z}
    padding = get_tree_padding(depth, nesting)

    return string(binary.op, " at ", show_location(X, Y, Z), '\n',
                  padding, "├── ", tree_show(binary.a, depth+1, nesting+1), '\n',
                  padding, "└── ", tree_show(binary.b, depth+1, nesting))
end

"Return a string representaion of a `MultiaryOperation` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(multiary::MultiaryOperation{X, Y, Z, N}, depth, nesting) where {X, Y, Z, N}
    padding = get_tree_padding(depth, nesting)

    out = string(multiary.op, " at ", show_location(X, Y, Z), '\n',
        ntuple(i -> padding * "├── " * tree_show(multiary.args[i], depth+1, nesting+1) * '\n', Val(N-1))...,
                    padding * "└── " * tree_show(multiary.args[N], depth+1, nesting)
                )
    return out
end

"Return a string representaion of a `Derivative` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(deriv::Derivative{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = get_tree_padding(depth, nesting)

    return string(deriv.∂, " at ", show_location(X, Y, Z), " via ", show_interp(deriv.▶), '\n',
                  padding, "└── ", tree_show(deriv.arg, depth+1, nesting))
end
