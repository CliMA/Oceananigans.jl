# Lid-driven cavity

The lid-driven cavity test problem has been used for a long time as a simple verification test for computational fluid
dynamics codes. First described by [Burggraf66](@cite), the fluid is contained in a square cavity with Dirchlet boundary
conditions on all four sides. The top wall moves with some velocity ``U`` while the other three walls are stationary.
The solution reaches a laminar steady-state whose properties can be compared with a huge amount of existing data. The
canonical database is given by [Ghia82](@cite) who report detailed information on the velocity fields as well as the
streamline and vorticity contours at various Reynolds numbers. More accurate data is reported by [Botella98](@cite),
[Erturk05](@cite), and [Bruneau06](@cite).
