struct UnaryOperation{X, Y, Z, A, O, G} <: AbstractOperation{X, Y, Z, G}
      op :: O
       a :: A
    grid :: G
    function UnaryOperation{X, Y, Z}(op, a) where {X, Y, Z}
        new{X, Y, Z, typeof(op), typeof(data(a)), typeof(a.grid)}(op, data(a), a.grid)
    end
end
