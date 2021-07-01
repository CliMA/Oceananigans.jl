# Contributors Guide

Thank you for considering contributing to Oceananigans! 

Feel free to ask us questions and chat with us at any time about any topic at all
by 

* [Opening a GitHub issue](https://github.com/CliMA/Oceananigans.jl/issues/new/choose)
 
* [Creating a GitHub discussion](https://github.com/CliMA/Oceananigans.jl/discussions/new)

* Sending a message to the [#oceananigans channel](https://julialang.slack.com/archives/C01D24C0CAH) on [Julia Slack](https://julialang.org/slack/).

We follow the [ColPrac guide](https://github.com/SciML/ColPrac) for collaborative
practices. We ask that new contributors read that guide before submitting a pull request.

## Creating issues

The simplest way to contribute to Oceananigans is to create or comment on issues and discussions.

The most useful bug reports:

* Provide an explicit code snippet --- not just a link --- that reproduces the bug in the latest tagged version of Oceananigans. This is sometimes called the ["minimal working example"](https://en.wikipedia.org/wiki/Minimal_working_example). Reducing bug-producing code to a minimal example can dramatically decrease the time it takes to resolve an issue.

* Paste the _entire_ error received when running the code snippet, even if it's unbelievably long.

* Use triple backticks (```` ``` ````) to enclose code snippets, and other [markdown formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax) to make your issue easy and quick to read.

* Report the Oceananigans version, Julia version, machine (especially if using a GPU) and any other possibly useful details of the computational environment in which the bug was created.

Discussions are recommended for asking questions about (for example) the user interface, implementation details, science, and life in general.

## But I want to _code_!

* New users help write Oceananigans code and documentation by [forking the Oceananigans repository](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks), [using git](https://guides.github.com/introduction/git-handbook/) to edit code and docs, and then creating a [pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork). Pull requests are reviewed by Oceananigans collaborators.

* A pull request can be merged once it is reviewed and approved by collaborators. If the pull request author has write access, they have the reponsibility of merging their pull request. Otherwise, Oceananigans.jl collabators will execute the merge with permission from the pull request author.

* Note: for small or minor changes (such as fixing a typo in documentation), the [GitHub editor](https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository) is super useful for forking and opening a pull request with a single click.

* Write your code with love and care. In particular, conform to existing Oceananigans style and formatting conventions. For example, we love verbose and explicit variable names, use `TitleCase` for types, `snake_case` for objects, and always.put.spaces.after.commas. For formatting decisions we loosely follow the [YASGuide](https://github.com/jrevels/YASGuide). It's worth few extra minutes of our time to leave future generations with well-written, readable code.

## What is a "collaborator" and how can I become one?

* Collaborators have permissions to review pull requests and  status allows a contributor to review pull requests in addition to opening them. Collaborators can also create branches in the main Oceananigans repository.

* We ask that new contributors try their hand at forking Oceananigans, and opening and merging a pull request before requesting collaborator status.

## What's a good way to start developing Oceananigans?

* Tackle an existing issue. We keep a list of [good first issues](https://github.com/CLiMA/Oceananigans.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  that are self-contained and suitable for a newcomer to try and work on.

* Try to run Oceananigans and play around with it to simulate your favorite
  fluids and ocean physics. If you run into any bugs/problems or find it difficult
  to use or understand, please open an issue!

* Write up an example or tutorial on how to do something useful with
  Oceananigans, like how to set up a new physical configuration.

* Improve documentation or comments if you found something hard to use.

* Implement a new feature if you need it to use Oceananigans.

If you're interested in working on something, let us know by commenting on
existing issues or by opening a new issue. This is to make sure no one else
is working on the same issue and so we can help and guide you in case there
is anything you need to know beforehand.

We also hang out on the #oceananigans channel on Julia Slack, which is a great
place to discuss anything Oceananigans-related, especially contributions! To
join the Julia Slack, go to [https://julialang.org/slack/](https://julialang.org/slack/).
