for op_string in ("UnaryOperation", "BinaryOperation", "MultiaryOperation", "Derivative")
    op = eval(Symbol(op_string))
    @eval begin
        operation_name(::$op) = $op_string
    end
end

Base.show(io::IO, operation::AbstractOperation) =
    print(io,
          operation_name(operation), " at ", show_location(operation), '\n',
          "├── grid: ", typeof(operation.grid), '\n',
          "│   ├── size: ", size(operation.grid), '\n',
          "│   └── domain: ", show_domain(operation.grid), '\n',
          "└── tree: ", "\n"^2, tree_show(operation, 0, 0))

"Return a representaion of number or function leaf within a tree visualization of an `AbstractOperation`."
tree_show(a::Union{Number, Function}, depth, nesting) = string(a)

"Fallback for displaying a leaf within a tree visualization of an `AbstractOperation`."
tree_show(a, depth, nesting) = short_show(a) # fallback

"Returns a string corresponding to padding characters for a tree visualization of an `AbstractOperation`."
get_tree_padding(depth, nesting) = "    "^(depth-nesting) * "│   "^nesting

"Return a string representaion of a `UnaryOperation` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(unary::UnaryOperation{X, Y, Z}, depth, nesting)  where {X, Y, Z}
    padding = get_tree_padding(depth, nesting)

    return string(unary.op, " at ", show_location(X, Y, Z), " via ", unary.▶, '\n',
                  padding, "└── ", tree_show(unary.arg, depth+1, nesting))
end

"Return a string representaion of a `BinaryOperation` leaf within a tree visualization of an `AbstractOperation`."
function tree_show(binary::BinaryOperation{X, Y, Z}, depth, nesting) where {X, Y, Z}
    padding = get_tree_padding(depth, nesting)

    return string(binary.op, " at ", show_location(X, Y, Z), " via ", binary.▶op, '\n',
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

    return string(deriv.∂, " at ", show_location(X, Y, Z), " via ", deriv.▶, '\n',
                  padding, "└── ", tree_show(deriv.arg, depth+1, nesting))
end
