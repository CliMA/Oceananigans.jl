@inline g_x(g::Number) = 0
@inline g_y(g::Number) = 0
@inline g_z(g::Number) = g

@inline g_x(g::Tuple) = g[1]
@inline g_y(g::Tuple) = g[2]
@inline g_z(g::Tuple) = g[3]
