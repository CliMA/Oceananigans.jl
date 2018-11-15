const N = (4, 6, 8)
const L = (2π, 3π, 5π)

function test_initfield()
  g = RegularCartesianGrid(N, L)
  a = CellField(g)
  b = FaceField(g, :x)
  c = FaceField(g, :y)
  d = FaceField(g, :z)
  Nxface = (N[1]+1, N[2], N[3])
  Nyface = (N[1], N[2]+1, N[3])
  Nzface = (N[1], N[2], N[3]+1)
  ( size(a) == N && 
    size(b) == Nxface &&
    size(c) == Nyface &&
    size(d) == Nzface )
end

function test_setfield()
  g = RegularCartesianGrid(N, L)
  a = CellField(g)
  set!(a, 2)
  a.data == 2*ones(N)
end

function test_addfield()
  g = RegularCartesianGrid(N, L)
  u = CellField(g)
  v = CellField(g)
  set!(u, 2)
  set!(v, 4)
  w = u + v
  wanswer = 6 * ones(N)
  w.data == wanswer
end
