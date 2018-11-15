# OceanDispatch.jl
A fast non-hydrostatic n-dimensional ocean model based on the MITgcm algorithm in Julia. The plan is to make it useful as a large eddy simulation (LES) model or as a fast super-parameterization to be embedded (or *dispatched*) within a global ocean model. As an embedded model it could resolve the sub-grid scale physics and communicate their effects back to the global model or act as a source of training data for statistical learning algorithms.

It may end up as a general-purpose ocean model that can be used in a hydrostatic or non-hydrostatic configuration with a friendly and intuitive user interface. Thanks high-level zero-cost abstractions (such as multiple *dispatch*) in Julia we think we can make the model look and behave the same no matter the dimension or grid of the underlying simulation.

 Just found out about the [Zen of Python](https://www.python.org/dev/peps/pep-0020/) which might be a great role model for this package
```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Now is better than never.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
```
