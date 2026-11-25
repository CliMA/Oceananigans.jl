const fused_multiply_add_choice = @load_preference("fused_multiply_add", "muladd")

replace_muladd(x, replacement) = x
replace_muladd(x::Symbol, replacement) = x === :muladd ? replacement : x
replace_muladd(x::Function, replacement) = x === Base.muladd ? replacement : x
replace_muladd(x::GlobalRef, replacement) = x.name === :muladd ? replacement : x
replace_muladd(ex::Expr, replacement) = Expr(ex.head, map(arg -> replace_muladd(arg, replacement), ex.args)...)

function apply_fused_multiply_add_choice(expression, choice)
    choice == "muladd" && return expression
    choice == "fma" && return replace_muladd(expression, GlobalRef(Base, :fma))
    error("The fused_multiply_add preference must be \"muladd\" or \"fma\", got \"$choice\".")
end

"""
    @fused_muladd expression

`MuladdMacro.@muladd`, with the `muladd` calls it generates either kept (the `fused_multiply_add` preference of
Oceananigans is `"muladd"`, the default) or replaced by `fma` (`"fma"`).
"""
macro fused_muladd(ex)
    expanded = macroexpand(__module__, Expr(:macrocall, GlobalRef(MuladdMacro, Symbol("@muladd")), __source__, ex))
    return esc(apply_fused_multiply_add_choice(expanded, fused_multiply_add_choice))
end
