<!-- Title -->
<h1 align="center">
  Oceananigans.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌊 快速且友好的面向海洋的 Julia 软件，用于在 CPU 和 GPU 上在笛卡尔和平壳域中模拟不可压缩流体动力学。更多信息见文档：https://clima.github.io/OceananigansDocumentation/</strong>
</p>

<!-- Information badges -->
<p align="center">
  <a href="https://github.com/CliMA/Oceananigans.jl/releases">
    <img alt="GitHub tag (latest SemVer pre-release)" src="https://img.shields.io/github/v/tag/CliMA/Oceananigans.jl?include_prereleases&label=latest%20version&logo=github&sort=semver&style=flat-square">
  </a>
  <a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
  </a>
  <a href="https://github.com/CliMA/Oceananigans.jl/discussions">
    <img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
  </a>
  <a href="https://github.com/SciML/ColPrac">
    <img alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" src="https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet?style=flat-square">
  </a>
  <a href="https://doi.org/10.21105/joss.02018">
    <img alt="JOSS" src="https://joss.theoj.org/papers/10.21105/joss.02018/status.svg">
  </a>
</p>

<!-- Documentation and downloads -->
<!-- counts downloads from individual IPs excluding bots (eg, CI) -->
<!-- see https://discourse.julialang.org/t/announcing-package-download-stats/69073 -->
<p align="center">
  <a href="https://clima.github.io/OceananigansDocumentation/stable">
    <img alt="Stable documentation" src="https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square">
  </a>
  <a href="https://clima.github.io/OceananigansDocumentation/dev">
    <img alt="Development documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
  </a>
  <a href="https://juliapkgstats.com/pkg/Oceananigans">
    <img alt="Downloads per month" src="https://img.shields.io/badge/Downloads-julia%20package-brightgreen?style=flat-square">
  </a>
  <a href="https://juliapkgstats.com/pkg/Oceananigans">
    <img alt="Total downloads" src="https://img.shields.io/badge/Total%20Downloads-juliapkgstats-blue?style=flat-square">
  </a>
</p>

Oceananigans 是一个快速、友好且灵活的软件包，用于在 CPU 和 GPU 上对非静水和静水 Boussinesq 方程进行有限体积模拟。
它可以在 GPU 上运行（哇，[很快！](https://doi.org/10.1029/2024MS004465)），不过我们相信 Oceananigans 最吸引人的地方是其超灵活的用户界面：它让简单的模拟变得容易，并且使复杂、富有创造力的模拟成为可能。
Oceananigans 的开发由社区驱动，贡献者来自学术界和工业界。
测试基础设施由 [atdepth](https://www.atdepth.org) 和 [气候建模联盟（Climate Modeling Alliance）](https://clima.caltech.edu) 提供。

## 目录

- [安装说明](#安装说明)
- [运行你的第一个模型](#运行你的第一个模型)
- [Oceananigans 知识库](#oceananigans-知识库)
- [引用及传播](#引用及传播)
- [贡献](#贡献)
- [视频演示（Movies）](#视频演示movies)
  - [深对流（Deep convection）](#深对流deep-convection)
  - [自由对流（Free convection）](#自由对流free-convection)
  - [海面风应力（Winds blowing over the ocean）](#海面风应力winds-blowing-over-the-ocean)
  - [带风应力的自由对流（Free convection with wind stress）](#带风应力的自由对流free-convection-with-wind-stress)
- [性能基准](#性能基准)

## 安装说明

Oceananigans 是一个 [注册的 Julia 包](https://julialang.org/packages/)。因此要安装它：

1. [下载 Julia](https://julialang.org/downloads/)（版本 1.9 或更高）。

2. 启动 Julia 并输入：

```julia
julia> using Pkg

julia> Pkg.add("Oceananigans")
```

这将安装与你当前环境兼容的最新版本。
别忘了小心查看你安装的是哪个 Oceananigans：

```julia
julia> Pkg.status("Oceananigans")
```

## 运行你的第一个模型

让我们运行一个二维、水平周期的湍流模拟，使用 128² 个有限体积单元，模拟 4 个无量纲时间单位：

```julia
using Oceananigans
grid = RectilinearGrid(CPU(), size=(128, 128), x=(0, 2π), y=(0, 2π), topology=(Periodic, Periodic, Flat))
model = NonhydrostaticModel(; grid, advection=WENO())
ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)
simulation = Simulation(model; Δt=0.01, stop_time=4)
run!(simulation)
```

另外，将 `CPU()` 改为 `GPU()` 可以让上述代码在支持 CUDA 的 Nvidia GPU 上运行。

深入阅读 [文档](https://clima.github.io/OceananigansDocumentation/stable/) 来获取更多代码示例和教程。
在下方，你会看到来自 GPU 模拟的视频以及 CPU 与 GPU 的[性能基准](https://github.com/clima/Oceananigans.jl#performance-benchmarks)。

## Oceananigans 知识库

它内容丰富，包含：

* [文档](https://clima.github.io/OceananigansDocumentation/stable)，提供：
    * Oceananigans 示例脚本，
    * 介绍关键 Oceananigans 对象与函数的教程，
    * 关于 Oceananigans 基于有限体积数值方法的说明，
    * Oceananigans 模型所求解动力方程的详细描述，和
    * 所有面向用户的 Oceananigans 对象与函数的 API 文档。
* [Oceananigans 的 GitHub 讨论区（Discussions）](https://github.com/CliMA/Oceananigans.jl/discussions)，涵盖话题例如：
    * ["计算科学（Computational science）"](https://github.com/CliMA/Oceananigans.jl/discussions/categories/computational-science)，或关于如何在 Oceananigans 中进行科学计算并设置数值模拟的讨论，
    * ["实验性特性（Experimental features）"](https://github.com/CliMA/Oceananigans.jl/discussions?discussions_q=experimental+features)，讨论新功能和稀疏文档的特性，适合喜欢探索的用户。

    如果你有问题或想讨论任何事情，请随时 [开启新的讨论](https://github.com/CliMA/Oceananigans.jl/discussions/new).
* [Oceananigans 维基（wiki）](https://github.com/CliMA/Oceananigans.jl/wiki) 包含关于 [开始使用 Julia 的实用提示](https://github.com/CliMA/Oceananigans.jl/wiki/Installation-and-getting-started)。
* Julia Slack 上的 `#oceananigans` 频道（https://julialang.org/slack/），可以访问 Oceananigans 社区中成员的“机构知识”。
* [Issues](https://github.com/CliMA/Oceananigans.jl/issues) 和 [Pull Requests](https://github.com/CliMA/Oceananigans.jl/pulls) 也包含了我们发现的问题及解决方案的许多信息。

## 引用及传播

如果你在研究、教学或娱乐中使用了 Oceananigans，我们的社区会非常感激你以名称对 Oceananigans 进行引用。

社区已经发表了若干描述 Oceananigans 开发的文章，包括一篇最近提交到 Journal of Advances in Modeling Earth Systems（JAMES）的预印本，概述了 Oceananigans 的总体设计与特性：

> “High-level, high-resolution ocean modeling at all scales with Oceananigans”
>
> G. L. Wagner, S. Silvestri, N. C. Constantinou, A. Ramadhan, J.-M. Campin, C. Hill, T. Chor, J. Strong-Wright, X. K. Lee, F. Poulin, A. Souza, K. J. Burns, J. Marshall, R. Ferrari
>
> Submitted to the Journal of Advances in Modeling Earth Systems, arXiv:2502.14148

<details><summary>bibtex</summary>
  <pre><code>@article{Oceananigans-overview-paper-2025,
  title = {{High-level, high-resolution ocean modeling at all scales with Oceananigans}},
  author = {G. L. Wagner and S. Silvestri and N. C. Constantinou and A. Ramadhan and J.-M. Campin and C. Hill and T. Chor and J. Strong-Wright and X. K. Lee and F. Poulin and A. Souza and K. J. Burns and J. Marshall and R. Ferrari},
  journal = {arXiv preprint},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2502.14148},
  doi = {10.48550/arXiv.2502.14148},
  notes = {submitted to the Journal of Advances in Modeling Earth Systems},
}
</code></pre>
</details>

请在发表中引用上述概述性论文（如适用）。

我们还发表/提交了若干模型开发相关论文。如果你使用了这些论文中所描述的功能，请引用它们；如果你在 Oceananigans 中开发了新功能并在论文中描述，请开一个 Pull Request 将其添加到列表中：

* Silvestri et al., "A new WENO-Based momentum advection scheme for simulations of ocean mesoscale turbulence" (https://doi.org/10.1029/2023MS004130).

  该文描述了 `WENOVectorInvariant()` 平流方案的开发，可用于 `HydrostaticFreeSurfaceModel` 的动量平流。

* Silvestri et al., "A GPU-based ocean dynamic core for routine mesoscale-resolving climate simulations" (https://doi.org/10.1029/2024MS004465).

  该文描述了对 `HydrostaticFreeSurfaceModel` 算法的优化，包括在分布式多 GPU 架构中实现 `SplitExplicitFreeSurface` 算法，使得近全球尺度、约 O(10 km) 网格的模拟能在 16–20 个节点上以大约 10 SYPD（simulated years per day）运行。

* Wagner et al., "Formulation and calibration of CATKE, a one-equation parameterization for microscale ocean mixing" (https://doi.org/10.1029/2024MS004522).

  该文描述了 `CATKEVerticalDiffusivity()` 的开发及其自动校准流程，并展示了与 `TKEDissipationVerticalDiffusivity`（即 k-ε）相关的结果。

* Ramadhan et al., "Oceananigans.jl: Fast and friendly geophysical fluid dynamics on GPUs" (https://doi.org/10.21105/joss.02018).

  这篇 JOSS 文章描述了 Oceananigans 早期版本中 `NonhydrostaticModel` 的实现。

我们维护着一个 [使用 Oceananigans.jl 的论文列表](https://clima.github.io/OceananigansDocumentation/stable/#Papers-and-preprints-using-Oceananigans)。
如果你有使用 Oceananigans 的成果并希望被列入，请提交 Pull Request 或告知我们。

## 贡献

如果你有兴趣为 Oceananigans 的开发做贡献，不论贡献大小，我们都非常欢迎！
如果你想开发新特性，或作为开源新手寻找合适的任务，请 [开启一个讨论](https://github.com/CliMA/Oceananigans.jl/discussions) 以便我们帮助你入手。

更多信息请查看我们的 [贡献者指南](https://clima.github.io/OceananigansDocumentation/stable/contributing/)。

## 视频演示（Movies）

### 深对流（Deep convection）
视频链接：https://www.youtube.com/watch?v=kpUrxnKKMjI

[![深对流缩略图](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/surface_temp_3d_00130_halfsize.png)](https://www.youtube.com/watch?v=kpUrxnKKMjI)

### 自由对流（Free convection）
视频链接：https://www.youtube.com/watch?v=yq4op9h3xcU

[![自由对流缩略图](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/free_convection_0956.png)](https://www.youtube.com/watch?v=yq4op9h3xcU)

### 海面风应力（Winds blowing over the ocean）
视频链接：https://www.youtube.com/watch?v=IRncfbvuiy8

[![风应力缩略图](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/wind_stress_0400.png)](https://www.youtube.com/watch?v=IRncfbvuiy8)

### 带风应力的自由对流（Free convection with wind stress）
视频链接：https://www.youtube.com/watch?v=ob6OMQgPfI4

[![带风应力的自由对流缩略图](https://raw.githubusercontent.com/ali-ramadhan/ali-ramadhan.Github.io/master/img/wind_stress_unstable_7500.png)](https://www.youtube.com/watch?v=ob6OMQgPfI4)

## 性能基准

我们进行了一些性能基准测试（参见文档中的 [性能基准](https://clima.github.io/OceananigansDocumentation/stable/appendix/benchmarks/) 部分）来评估不同配置下 Oceananigans 的性能。

为了充分利用或完全饱和像 Nvidia Tesla V100 或 Titan V 这样的 GPU 的计算能力，模型的网格点数应当在大约 ~10,000,000（约 1000 万）点或更多。

有时候反直觉地使用 `Float32` 比 `Float64` 更慢。这很可能是由于类型不匹配导致的性能损失（浮点在 32 位和 64 位之间需要转换），这是需要细致处理的问题。由于其它瓶颈（比如内存访问和 GPU 寄存器压力），`Float32` 模型可能不会带来显著的加速，主要优点通常是内存占用更低（大约小一倍）。

![性能基准图示](https://user-images.githubusercontent.com/20099589/89906791-d2c85b00-dbb9-11ea-969a-4b8db2c31680.png)
