# Clock
The clock holds the current iteration number and time. By default the model starts at iteration number 0 and time 0
```@example
using Oceananigans # hide
clock = Clock(0.0, 0)
```
but can be modified if you wish to start the model clock at some other time. If you want iteration 0 to correspond to
$t = 3600$ seconds, then you can construct
```@example
using Oceananigans # hide
clock = Clock(3600.0, 0)
```
and pass it to the model.
