### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c86f289f-3a93-4299-863d-7cd2d2235489
using LinearAlgebra

# ╔═╡ 9b54663f-3886-43aa-a6da-9cf84aeedb4d
using Random # Already in the language, you are just accessing the namespace

# ╔═╡ 7bd3f2ea-efc3-46d1-a53f-de0e7827459d
begin
	import PlutoUI
	PlutoUI.TableOfContents(title="Intro to Julia")
end

# ╔═╡ 0216ddd0-c95a-11eb-202a-b5ac98989c51
md"""
# Julia and Modern Machine Learning

Looking at the Julia's current surge in popularity in scientific fields, it makes sense to start considering Julia for doing researching in the machine learning.

I will be giving a high-level tutorial on what you need to know about Julia for our course. The goals of this tutorial are: 
- to get you set up to use Julia for the purposes of the UofA ML Undergrad Courses,
- introduce some core aspects of how to use Julia, 
- point to key packages to take advantage of, 
- and give resources for learning more. 

This tutorial is written in Pluto notebooks, a new take on what an interactive notebook can be built from the ground up in Julia.


"""

# ╔═╡ 3221dfc7-7ddc-40d1-9ec8-e6d2f61feddf
md"""
### Why Julia

Julia is a modern language with numerics at its core. It is abstract and flexible like python, numerically focused like Matlab, and can be optimized to be as fast as c/c++/fortran. It is also the first language to seriously use multiple dispatch as a core design philosophy. While this may be hard to get used to when coming from OO languages, the Julia userbase has noticed some interesting properties suggesting its effectiveness in code reuse [link](https://www.youtube.com/watch?v=kc9HwsxE1OY). Julia also has built-in utilities for threading (go-style coroutines, and pthread style for loops), multi-processing, and efforts for language-wide auto differentiation tools with a unified and extensible approach. There is also support for GPU computations, including writing your own GPU kernels.


You may be asking yourself, why should we think about using Julia as Python is ubiquious in the field? I won't try and convince you in this notebook, but here are a few reasons I have for using Julia:

* Mulitple dispatch can often make code easier to use and extend as compared to OOP.
* Arrays are well thought about in the base language, so there is uniformity in design principles across numeric arrays, arrays for generic data, and arrays on specialized hardware (like GPUs). Core linear algebra is also a priority and a part of Base.
* Solves the two language problem: code can be effiecient or easy to read (and often both!) all in julia, so there is little need to turn to c or fortran for really efficient code. Julia is also a part of the exclusive petaflop club: https://www.avenga.com/magazine/julia-programming-language/. 
* Threads and Multi-processing are both a part of the base language and easy to use.

These are only a few reasons, and more extensive lists can be found elsewhere. If you have further questions you can ask Matt Schlegel about why he uses Julia in his ML/RL research and doesn't see a return to Python.
"""

# ╔═╡ 6b7d1012-e2d1-48a2-9e75-eaaf7ebc2a72
md"""
### What about Python?

Julia gives you the tools and flexibility to work at an abstract level (like Python) with the ability to work at a low level to optimize numerical code (like C/C++). While Python lets you write high-level abstract code, all the optimized numerical code is written in C/C++. This means operations not supported by NumPy will either need to be written in C or will be slow. Python can also be quite error-prone for new users, as certain operations are legal but not what you intended (e.g., dot product versus element-wise product) and certain language features can cause the code to become very slow (e.g., for loops). While some of these issues are being actively tackled by projects such as Numba and Cython, third-party package developers need to have explicit buy-in to these systems and develop code with these in mind. Chris Rackauckas has an excellent blog post discussing the core limitations to these approaches [link](https://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/). 

This is not to say Python is not an excellent language, it is and has become quite popular becuase of the trade-offs it makes. It is hard to predict if Julia will become a language of choice over other common languages for data analysis (Python, R, Matlab), but it has the potential to become widely used and has a growing user base, and has the potential to kick the machine [learning field out of its design run](https://dl.acm.org/doi/10.1145/3317550.3321441) caused by monolithic highly optimized kernels.
"""

# ╔═╡ 937863e8-7305-4639-a3dc-3924621ac5d3
md"""
# Design patterns

While I won't go into detail about design patterns which emerge from Julia's multiple dispatch and typing system, you should read [this blog](https://www.stochasticlifestyle.com/type-dispatch-design-post-object-oriented-programming-julia/) by Christopher Rackauckas (who is an active user of the language doing research in applying ML/AI methods to Scientific pursuits). 

I've noticed after introducing several students to Julia the hardest hurdle is understanding how to organize code. This is not new, and when faced with any new paradigm it can be daunting to try and understand how things are organized. Unlike other modern paradigms like OOP there is very little in the way of patterns that need to be known ([think gang of four](https://en.wikipedia.org/wiki/Design_Patterns)). This is a property of a multiple dispatch language and the seperation of functions/methods from types. You can see more of the neat properties of multiple dispatch in [this video](https://www.youtube.com/watch?v=kc9HwsxE1OY).

"""

# ╔═╡ 2fe2603e-8404-4cb0-a9c8-ed7497a88db2
md"""
# Other resources
"""

# ╔═╡ 15f526ae-8415-470c-aaf3-afb5603428db
md"""
### Good practices

- [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- [Differences from other Languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/)
- [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/)

"""

# ╔═╡ 8df4cb0f-cbdb-452f-a504-4e577e26999d
md"""
### Useful packages and projects

- [Pluto](https://github.com/fonsp/Pluto.jl): reactive notebooks written in and for julia
- [IJulia](https://github.com/JuliaLang/IJulia.jl): jupyter kernel
- [Revise](https://timholy.github.io/Revise.jl/stable/): for command line development
- [Julia for VSCode](https://www.julia-vscode.org): an editor outside of pluto/jupyter
- [Plots](http://docs.juliaplots.org/latest/): an extensive plotting package
- [Flux](https://fluxml.ai): neural network package

and many more...

"""

# ╔═╡ 0d84dfe5-ebb8-43fd-8164-2c2c9c06db4b
md"""
### Other resources for learning julia

- [Julia for Beginners](https://www.youtube.com/watch?v=ub3tqCWZmo4)
- [Documentation](https://docs.julialang.org/en/v1/)
- [Tutorials on the Julia website](https://julialang.org/learning/tutorials/)

"""

# ╔═╡ c0c73d7e-488d-440e-88ae-434f923efa2d
md"""
### Hurdles and known sharp edges

So far we've discussed several positive aspects of julia, but like with any language there are  edges where the design of the language can be frustrating. While we won't discuss these here, a great list has been compiled [here](https://viralinstruction.com/posts/badjulia/).
"""

# ╔═╡ 96dcdb5a-7733-411a-94e1-fc3a9d22fcf3
md"""
# Basics


In this section, we will be going over the basics of using the julia language. This should be enough for most of what you will need for the machine learning course.


- variables
- variable scope
- types
- arrays and other collections
- loops
- structs and data
- linear algebra/mathematical operations
- Style guide and other tips/recommendations


"""

# ╔═╡ 8aeacbc2-252e-4428-9884-be6ef481bfca
md"""
## Variables

Variables in julia are created and assigned like most dynamic languages. In julia, every variable is given a type, and if that variable remains the same type throughout its scope it is considered type-stable. You can inspect the typeof of a variable using `typeof`, as seen below.


"""

# ╔═╡ 0c146b61-4322-4d1b-8080-36a9c7b430a9
x = 10.0

# ╔═╡ 5983b16e-34d7-4b44-9236-ab9b7fa77d5f
typeof(x)

# ╔═╡ a62e4f22-6b68-40dd-96cb-8ccc970eeb3b
md"""
## Variable Scope in Julia/Pluto

Variables are scopped in Julia/Pluto in a similar way to Python/Jupyter. Variables defined in a cell are global and available across all the cells. This can sometimes cause problems in Jupyter notebooks where variables can be overwritten unknowingly. In Pluto, you are unable to re-use a variable by accident (see below for `y`), Pluto doesn't know which definition you mean and throws an error for both.
"""

# ╔═╡ 6e65025b-e625-4445-8dca-99e782d6c2e2
md"""
While a win for reproducibility and simplicity, this can be annoying when you have throw-away variables you wan to use/name. Fortunately, there is a way around this!
"""

# ╔═╡ 75cb3f61-859e-428b-96f6-1c358287baac
md"""
### Begin and Let Blocks

The begin and let blocks are both used to make multi-statement cells in our notebook.

The begin block makes all the variables available at the global scope:
"""

# ╔═╡ 08aa01ed-eeb5-4296-a4b2-027582c7eeb8
begin
	my_global_x = 10
	# do other stuff
	my_global_x *= 20
end

# ╔═╡ f5f4e8dc-d770-477a-94d7-a5468d263d5a
my_global_x

# ╔═╡ 04daba2b-17e2-4468-9a56-822e6358ea74
md"""
A let block on the other hand keeps all its variables local. Notice how `my_local_x` is not available to the cell right after the let block.
"""

# ╔═╡ 5ca1c7ef-dc60-474f-8a2c-03f8ae6143df
let
	my_local_x = 1020
	# do other stuff
	my_local_x *= 1029
end

# ╔═╡ 92af2fb1-a271-4410-8630-41db7559016a
my_local_x

# ╔═╡ 40046c3a-2049-46e0-89a0-4d86a5d9345f
md"""
Both of these work within functions and other scopes. You can also consider most scopes (i.e. functions, loops, control flow statements) as let blocks.
"""

# ╔═╡ 344ce300-1964-41c1-b19e-5c18b00d12f5
md"""
## Types

Types are an important part of Julia's ecosystem. There are many primative types, which are represented as a collection of bits, including `Int64`, `Float32`, `Float64`, and many more. You can see a full list [here](https://docs.julialang.org/en/v1/manual/types/#Primitive-Types). 

Types are placed into a type hierarchy, where leaf nodes of the hierarchy are objects you will be interacting with. For example, `Float64` is an `AbstractFloat` which is a `Number`. You can inspect the type of a variable with the functions `isa` and `<:`. Below you can see their use, note how `isa` works with the variable while `<:` works with a type. You will likely not have to use these a lot in this course. 
"""

# ╔═╡ 14d445fd-b7be-4153-a550-4b486e0b680e
x isa Float64, x isa AbstractFloat, x isa Number

# ╔═╡ da49090c-50ef-4fe4-9eaa-005c2d65e002
typeof(x) <: Float64, typeof(x) <: AbstractFloat, typeof(x) <: Number

# ╔═╡ da024556-51dd-4032-9a70-c346bb442e08
md"""
### Arrays

Every array will have two parametric types associated with it. You can create an array of `Float64` with three dimensions as below. When we inspect the type you will see `Array{Float64, 3}` the first component in the curly braces is the element type of the array, and the second is the number of the dimensions the array has. A vector has 1 dimension, and a matrix has 2 dimensions. 

If you change x_arr to be a matrix by removing the `2` you will see the type change to `Matrix{Float64} (alias for Array{Float64, 2})`. `Matrix` is a convenient name for 2d arrays, but their underlying type is exactly the same. 
"""

# ╔═╡ 4313df31-46ab-4d07-ac22-2970e25e44cf
x_arr = rand(4, 3, 2)

# ╔═╡ 10a0bd50-eae0-40ef-93e1-fa6566a53431
typeof(x_arr)

# ╔═╡ baf0edb4-81bc-439d-bdcd-e5f6b9c135fe
typeof(zeros(10, 3))

# ╔═╡ 2d0303f1-985d-440e-9fc4-8ef1e711f47f
md"""
You can get the element type of an array with the following
"""

# ╔═╡ 4cf0e24d-17a7-4d02-990f-2737c9fad982
eltype(x_arr)

# ╔═╡ a87690b7-3486-4089-a5ad-91cfe9b34181
md"""
Julia as one-based indexing. To access an element of an array simply use square brackets as the following:
"""

# ╔═╡ cfc62088-fb69-429d-aa26-56dbe51430c5
x_vec = rand(10)

# ╔═╡ 566c53c5-c411-45b0-8fe8-c8beff4b372d
x_vec[1]

# ╔═╡ 394ed1d5-b0e4-40ce-9528-1d46d431a786
md"""
And you can assign to this array
"""

# ╔═╡ 84eadf22-a146-43f9-bb92-9da84059e11a
x_vec[1] = 1

# ╔═╡ 186004b2-3bfb-4fc9-9da5-d39b155542ab
x_vec

# ╔═╡ d1605fa3-4b02-435e-afb6-1f81dcb4f378
md"""
Finally, you can get a slice from a multi-dimensional array using `:`. Note how this returned vector is a column vector, and not a `1×1×2` strand.
"""

# ╔═╡ 244cfc1b-707b-4fd7-b435-2cc3ee5075dc
x_arr[1, 1, :]

# ╔═╡ 7fee83f9-a41c-47d9-b888-34e0f9bb6e59
md"""
and you can assign to these strands:
"""

# ╔═╡ 6c1a3bbe-b465-4088-b76a-bbdd854a58f0
begin
	x_arr[1, 1, :] .= 1
	x_arr
end

# ╔═╡ 70d30b70-4adf-4fcf-bb78-6e731fd14012
begin
	x_arr[1, 2, :] .= [2, 3]
	x_arr
end

# ╔═╡ 14e2c8c0-82ea-4c14-b833-c4466d1b7d5d
md"""
Julia has a fully featured set of linear algebra operations built into the language which you can see later in this tutorial.
"""

# ╔═╡ 8c0584c5-7754-4a0e-8e75-2ec6f7a42a1b
md"""
### Other collection types

There are several other collection types which will be useful.
"""

# ╔═╡ 3314a10a-bba6-4552-9de5-c7a843325f7b
md"""
#### Dictionaries

A dictionary is a collection which encodes key=>value pairs using a hash table. Dictionaries created with `Dict()` accept any types for the keys and values. You can create dicts with more specifict types using `Dict{String, Int}()` which will have keys as `String` and values as `Int`. 
"""

# ╔═╡ 548f9daa-434a-4786-99bd-48ce5ea782e4
d = Dict()

# ╔═╡ 08e36c0f-39c7-402e-9e9b-8582b7e0965e
d["hello"] = "world"

# ╔═╡ d8533b95-764a-44fc-bf79-6ee16a0ca8dc
d["name"] = "Matthew"

# ╔═╡ 876fe161-7856-434a-993f-adc0ce6e6ccb
d

# ╔═╡ 6fe86984-ca31-4dac-afb8-1dbab8cfb080
typeof(d)

# ╔═╡ 82b85f5c-a69a-4abe-98a4-8ca9ec16b2c1
md"You can get the keys from a dictionary with the following"

# ╔═╡ bc411307-1e1b-41a8-8639-03e0adadb5e6
keys(d)

# ╔═╡ 6fb6b701-4c9a-4dc7-80c3-44bd600e3a26
md"""
#### Tuples

Tuples encode a finite set of elements of any type. They are constructed as seen below, and have types which depend on the types of all of its elements.
"""

# ╔═╡ a2512e72-6c53-4503-84cd-58f50f34d6e9
tpl = (1, "1", '1', 1.0, 1f0)

# ╔═╡ 0ac59e50-84d6-4e29-859a-a94d47cf42c2
typeof(tpl)

# ╔═╡ 960aca6b-90e9-4249-aefe-9f4bbce64cc6
md"Compared to an array"

# ╔═╡ 23b70d5b-5268-4f12-b12b-e3940e3d08fc
arr = [1, "1", '1', 1.0, 1f0]

# ╔═╡ e0fc96d6-5b27-4baa-8d0b-663e67c3617a
typeof(arr)

# ╔═╡ 486b3b19-af29-42c2-aebb-43b360e0a66f
md"**Note**: Unlike an array, you cannot assign to an element of a tuple"

# ╔═╡ 90f24d6a-b842-4cb1-be6e-66b78faa57fc
tpl[end] = 2f0

# ╔═╡ 31b53c35-91c3-4c4e-86e0-a48bbd7aa8bf
md"""
#### Named Tuples

Named tuples are just like tuples except each element has a name.
"""

# ╔═╡ b2524a17-ef89-4ffe-8ab9-bb5034dd1803
ntpl = (a=1, b=2)

# ╔═╡ 4b038409-c4ab-4d76-b9ee-90880917ddf5
ntpl.a

# ╔═╡ 2263d894-745a-46a2-a74f-de82231c9580
typeof(ntpl)

# ╔═╡ d98759f8-2e5b-416e-b1d9-da2fc932929a
md"""
## Control Flow

Below are the basics of control flow. This includes Loops and If-Else statements
"""

# ╔═╡ e89e2dde-db5d-43c0-a627-3761e3439b0b
md"""
### If-Else Statements

If-else statments work much like they do in other languages. Below is an example of if statements and if-else statements. If there is an if statement w/o an else it returns nothing by default. Like all julia blocks, the last statement will be returned automatically.
"""

# ╔═╡ 0cd3e086-2694-4ed5-a95e-82cc20c64b06
let
	i = 2
	if i < 3
		md"i=$(i) and is less than 3"
	end
end

# ╔═╡ 0268ec5c-191f-4389-9d07-0da8b213be6b
let
	i = 3
	if i < 3
		md"i=$(i) and is less than 3"
	else
		md"i=$(i) and is ``\geq`` to 3"
	end
end

# ╔═╡ 442f3ff7-fcdd-4329-98bc-da10a6c4d8eb
let
	i = 3
	if i < 3
		md"i=$(i) and is less than 3"
	elseif i == 3
		md"i=3"
	else
		md"i=$(i) and is ``>`` to 3"
	end
end

# ╔═╡ 4e5567a1-8fcb-49d4-baab-e3d1e21959fc
md"""
Another common pattern is to check for `nothing`. Unlike other languages a variable set to nothing is not assumed false and must be checked explicilty.
"""

# ╔═╡ 6c654de3-71e2-4263-ac1e-2b9efd477457
let
	my_var = nothing
	if isnothing(my_var)
		md"The variable is nothing!"
	else
		# Do Some work w/ my_var here
	end
end

# ╔═╡ dc2362c0-f1db-45f0-97aa-2bc5e859c88e
md"""
### Loops

Like most languages, loops come in two varieties: `for` and `while`. See the examples below.

"""

# ╔═╡ 5e99e46c-55bd-441a-885a-33134e45ea87
let
	i = 0
	for n in 1:10 # 1:10 creates a range containing all the numbers from 1 to 10.
		i += n
	end
	i
end

# ╔═╡ 95c06ad7-37ae-4765-8a18-414525d07421
let
	x = 0.0
	i = 0
	while x < 20.0
		x += rand()
		i += 1
	end
	i, x
end

# ╔═╡ e681c1a6-c839-4ca1-b2fd-8277b358a47d
md"""
We can even loop over the elements of various collections:
"""

# ╔═╡ 766ad54e-2e33-4cca-a0b0-12a1ebcbb12e
let
	X = rand(10, 4, 2)
	ret = 0.0
	for x in eachindex(X)
		ret += X[x]
	end
	ret
end

# ╔═╡ 50228280-277a-46cf-a2d9-d71a10196944
let
	X = rand(10, 4, 2)
	ret = 0.0
	for x in X
		ret += x
	end
	ret
end

# ╔═╡ 7f39f7c7-e98b-4054-ba3a-24e3c66d0cf7
let
	ret = String[]
	for k in keys(d)
		push!(ret, d[k])
	end
	ret
end

# ╔═╡ fa277da9-5355-4a96-929c-8bffb8b98e38
md"""
We cal also do list comprehensions.
"""

# ╔═╡ becf0487-51c6-47c0-aa4c-8e1578a879bd
[rand(i) for i in 1:5]

# ╔═╡ 0edfee8c-d35c-499c-95a2-67b4462fa087
md"""
## Structs and Data

Structs are how you can build your own composite types. For all of the assignments, the types will be provided for you. But it is good to know how these types are constructed.

For example:

```julia
struct MyType 
	n::Int
end
```

This types is named `MyType` with a member `n` which is typed as an integer. Each type comes with a default constructor, for example `MyType(1)`. 

You can access the data of a type using dot syntax (like c, python, and many other languages).

"""

# ╔═╡ 09dbc568-bfe1-49ea-b207-e983b7f80359
struct MyType 
	n::Int
end

# ╔═╡ 7103f6e0-ecad-4455-87e9-df8ef0eb7399
mt = MyType(1)

# ╔═╡ fab829c5-7421-4bc5-9d75-2ae512148e61
mt.n

# ╔═╡ dce3b77b-089b-4f6a-8f36-4d69af9d874e
md"**Note** we can't assign n using a struct, instead we need to make MyType a `mutable struct`"

# ╔═╡ 5fdb2101-6cb8-44a5-b51d-f82bc39cf2ba
mt.n = 2

# ╔═╡ 4b23db15-f28b-477e-8279-bfee0ba30c9f
md"To make a mutable type"

# ╔═╡ fef36c0d-4ace-4fe8-be21-4d8b6e05119a
mutable struct MyMutableType
	n::Int
end

# ╔═╡ f079db3f-582b-4df9-af7c-1baa8c6f79b8
mmt = MyMutableType(1)

# ╔═╡ 6f22651a-0c5c-48cb-b5f7-053c70d603b2
mmt.n = 2

# ╔═╡ db737d1c-7a1a-4206-8512-492a0dec0ae6
mmt

# ╔═╡ 98919df9-476b-4319-a296-ddd39455636a
md"If you have a mutable type composed in a non-mutable type you can still modify the data internal to the mutable member"

# ╔═╡ 58f6b213-066d-4abe-8ac9-ceb54e340d9b
let
	struct VectorWrapper
		arr::Vector{Float64}
	end
	VectorWrapper(n) = VectorWrapper(zeros(n)) # create a new constructor
end

# ╔═╡ 6e4f163b-cc0d-4e06-bcc9-a61ae58ee3e5
vw = VectorWrapper(10)

# ╔═╡ 509599bf-d60e-49e6-853a-dfd950684d4d
vw.arr[3] = 3

# ╔═╡ 40ebc8e0-b04d-4216-8e76-50284da2e988
vw.arr

# ╔═╡ 15217efb-26ec-43e8-b136-f089084bffa4
md"""
## Linear Algebra

Lets start simple, and perform some typical linear algebra operations using Julia to get a handle of the language. While these contain only a subset of what is baked into julia, you should be able to extrapolate to other operations you care about. In the code below only `dot` and `svd` are defined in `LinearAlgebra`.

"""

# ╔═╡ 9a097abc-b457-4a21-8073-bfa312eb1d68
md"""
---
### **An Aside on `using`**

There are two main ways to include code from packages in your own package/notebook. The first is with the `using` keyword. This will bring in the functions and types that have been marked for export by the package authors. It has been common practice to export names which are stable, and keep private internals or apis still inprogress.

The second way is through the `import` keyword. Doing:
```julia
import LinearAlgebra
```
will keep all the names defined by `LinearAlgebra` contained in its own namespace and accessed as `LinearAlgebra.svd`

You can also import specific functions using
```julia
import LinearAlgebra: dot, svd
```

Currently there is no `as` keyword (although this is being discussed actively), but you can re-name packages with `const LA = LinearAlgebra`.

---
"""

# ╔═╡ f5de53c4-8a33-40f3-9743-07d48830e3bb
a, b, s, M = rand(2), rand(2), rand(), rand(2,2)

# ╔═╡ 851249e9-3d96-4c6c-98c6-0bc96d29ae3d
md"##### Vector Addtion:"

# ╔═╡ 87a7a91d-553c-4e00-88eb-bfbb7da1a5ba
a + b

# ╔═╡ 505a7663-1e7e-4038-93c7-71c91ef6b2ab
md"##### Scalar multiplication:"

# ╔═╡ 112055b7-4776-4164-892d-685b098f101a
2a, 2*a, 2 .* a

# ╔═╡ 0871989c-298d-458a-a220-7f597a1a07f4
md"##### Inner product of vectors:"

# ╔═╡ 2fc9a30b-64bd-4980-910f-d8b8e5083232
a' * b, dot(a, b) # dot is from LinearAlgebra

# ╔═╡ 35b467e3-6bf5-4498-81ab-86376fd15101
md"##### Outer product of vectors:"

# ╔═╡ c4d7efd8-f99e-48f8-9892-38946eb89a17
a * b', a * transpose(b)

# ╔═╡ 8c3e53b8-037c-4f2a-9cc2-e39358216aa2
md"##### Matrix mulitplication of vectors:"

# ╔═╡ 0e5f5e35-53bb-4d17-bd2b-34bf8fab712d
M * a

# ╔═╡ 600ad0bd-0eb6-419a-a19f-d0c701910968
md"##### Matrix element wise product with outer product operation"

# ╔═╡ 6c6bec44-ed8e-499b-b1f4-c8753d07698e
M .* (a * b')

# ╔═╡ 48a4c4b9-e32f-4fbb-9e57-fe456973032c
md"##### Element wise vector multiplication (broadcasting):"

# ╔═╡ 783ae651-f8c4-4eca-81e2-57c6a807f46a
a .* b

# ╔═╡ 1d595759-9a4f-46f1-b030-3ab96df94fb0
md"##### Broadcast scalar-vector addtion"

# ╔═╡ 30d68c48-1683-4da2-98f5-e910038f0a45
a .+ s

# ╔═╡ 7352b072-5933-455c-b757-b8994b4dbe34
md"""
##### Much more

Along with these operations which are always available, there is a Base package `LinearAlgebra`

If you have a specfic linear algebra operation you want but can't find it in Base, you will need to explicitly load the `LinearAlgebra` package. This is already available to you in `Base`.

"""

# ╔═╡ 140391fb-60d1-472a-816f-c9a5800922b8
M_svd = svd(M)

# ╔═╡ a95d12ac-fe5b-4476-a572-4f5a44d6e9c5
all((M_svd.U * diagm(M_svd.S) * M_svd.Vt) .≈ M)

# ╔═╡ 331cc1ce-6d52-4919-8e2c-674847fc3f49
md"""
## Style Guide and tips

We will inherit as much from this [Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/) as possible, but much of this style guide is written for when we are using julia outside of a Pluto notebook. Because of this, you should follow the style of the notebook you are completing. Because you will be primarily focused on writing functions, this should be straight forward. 

Some tips:
- Make sure your variables are type-stable (i.e. try to make sure you don't re-use variables unnecesairily).
- Don't use any globals (all of your functions should be self contained).
- Use for loops if you are unsure about how broadcasting or other mathematical operation is working (loops are perfectly fine in julia).
- Make matrices (or multi-dim arrays) capital letters and vectors lower case.
- Use meaningful variable names. In julia you can type `\eta` and press tab to get the unicode symbol `η`, and julia supports many symbols. This is useful when writing implementing algorithms so we can use all the same symbols which are used. We will provide you with a table of symbols that will be useful to know.
- If you are unsure about how something works in julia or you want to look at how a function works, the best advice is to go look at the julia source. Most functions you will use are written in julia, and tend to be well documented and understandable. This is true in many packages as well.
"""

# ╔═╡ 5c081eb9-9989-4728-b3b6-f9577169d6c8
md"""
# Advanced topics

The following are not necessairy for the course, but provide some insight into some of the core concepts of the julia language and its design philosophy.

"""

# ╔═╡ b71dc229-42f3-457c-a8fb-f477d4b2e06c
md"""## Arrays are column major by default!

While this likely won't be of consequence for this course, it is important to note given Python and other languages have row major arrays. 

"""

# ╔═╡ 5ced5c9e-899f-44e9-b66e-90bd5348309b
a * M # this was meant to error

# ╔═╡ 611bef16-1eac-4fc4-844b-ef9e7ee23b4c
md"This can also make a difference for how we iterate through large arrays. Note the order of indicies in the loops below. `eachindex_iteration` provides a more general way to do what is provided in `column_major`."

# ╔═╡ 992d5019-a326-4492-9130-2911ed615a5a
function row_major(arr::Array{<:Number, 5})
	s = zero(eltype(arr)) # This sets s = 0.0 using the same type as arr.
	sze = size(arr)
	for i in 1:sze[1]
		for j in 1:sze[2]
			for k in 1:sze[3]
				for l in 1:sze[4]
					for m in 1:sze[5]
						s = s + arr[i, j, k, l, m]
					end
				end
			end
		end
	end
	s
end

# ╔═╡ 21960ac5-6816-4aca-b133-35d0406cfa2c
function column_major(arr::Array{<:Number, 5})
	s = zero(eltype(arr))
	sze = size(arr)
	for m in 1:sze[5]
		for l in 1:sze[4]
			for k in 1:sze[3]
				for j in 1:sze[2]
					for i in 1:sze[1]
						s += arr[i, j, k, l, m]
					end
				end
			end
		end
	end
	s
end

# ╔═╡ a4f25563-815b-44e6-a604-77ea1b55bccb
function eachindex_iteration(arr)
	s = zero(eltype(arr))
	for idx in eachindex(arr)
		s = s + arr[idx]
	end
	s
end

# ╔═╡ 106e8d97-5f74-4fc0-971d-8c3f1b0afeb1
md"""
## Benchmarking

Below we are benchmarking the functions we wrote above. Note the median and mean times for each function.
"""

# ╔═╡ f23b08b9-a1ee-48fd-a0b4-96460a30b1b0
let
	import BenchmarkTools: @benchmark # This benchmarks a method
	import PlutoUI: with_terminal, CheckBox
end

# ╔═╡ f0f8c60f-4a07-4edb-9b3d-c063e3190442
md"run example: $(@bind run_example CheckBox())"

# ╔═╡ 384d1dff-6f46-456d-839b-b889751e6f03
test_arr = run_example ? rand(Float64, 1000, 100, 32, 4, 4) : nothing;

# ╔═╡ 38477409-9d22-4ecc-9094-19eebf38afbf
check_test_arr(func::Function) = isnothing(test_arr) ? md"Click run example." : func()

# ╔═╡ 4d70beee-a91f-4ecb-a890-75b1db7bfae0
md"**Column Major Sum**"

# ╔═╡ ce200b5c-64b8-468d-b25a-fc630699e2c1
check_test_arr() do
	@benchmark column_major($test_arr)
end

# ╔═╡ 35dfe45d-d93e-428d-9fcd-36865a45468f
md"**Row Major Sum**"

# ╔═╡ a497e722-1291-479c-98cb-69dd93a6ac9c
check_test_arr() do
	@benchmark row_major($test_arr)
end

# ╔═╡ 179ebad2-0804-4b3b-a361-df84903d9bab
md"**Each index**"

# ╔═╡ 7d81633e-163e-40d2-a119-8cb2e507be4c
check_test_arr() do
	@benchmark eachindex_iteration($test_arr)
end

# ╔═╡ 4e1fbc51-7df4-4dab-8f9c-ad52cabea74a
md"""
##### Comparing to `sum`

Now even though we can write a fastish version of the sum operation using `eachindex` it is still often better to use julia's built in functionality. For instance if we compare the performance with the provided sum function in julia, we see a huge boost in performance (3 times on my computer!). 
"""

# ╔═╡ d4e2cb43-c0b9-4cd3-8dea-3973a20d9da8
check_test_arr() do
	@benchmark sum($test_arr)
end

# ╔═╡ 469fc6da-ab5e-4886-acea-2d94c6139494
md"""
Why is this? I thought julia was supposed to compile to effecient code?

It does! and it often has performance comparable with c++ if you allow for similar compiler optimizations. Unfortunately, writing efficient code today often goes beyond being compilable, and involves giving the compiler clues about what operation can be used. The good thing for us is the majority of `Base` is written in julia! You can see the `sum` implementation for `v1.6.2` (and more generally `mapreduce`) [here](https://github.com/JuliaLang/julia/blob/1b93d53fc4bb59350ada898038ed4de2994cce33/base/reduce.jl#L504). There is nothing stopping you from using the utilities used in this implementation in your own code!

If you look at the implementation, you will see the macro `@simd`. This tells the compiler that the loop can take advantage of `simd` instructions if your processor has them. We can even use this ourselves! This gets us most of the way to sum, with code that is effectively a for loop.
"""

# ╔═╡ 248d748b-8ce5-401d-94eb-214f440d4fc8
function eachindex_iteration_simd(arr)
	s = zero(eltype(arr))
	indicies = eachindex(arr)
	@simd for idx in indicies
		@inbounds a1 = arr[idx]
		s = s + a1
	end
	s
end

# ╔═╡ 0b21b833-f242-4fec-9c09-83d4b9d8628f
check_test_arr() do 
	@benchmark $eachindex_iteration_simd($test_arr)
end

# ╔═╡ 08ba814a-7a92-4be2-ae15-b94c62b60c2d
md"""
### Why not `@simd` by default?

When using simd the order the loop is executed in can be arbitrary, unlike without simd. This has to do with how things are vectorized. This means the answer can vary with difference orders of the sum. It is also hard to analyze the internals of a loop to make sure they follow the requirements to make sure `@simd` is safe (see the pluto docs for `@simd` or type `?@simd` in the julia repl).

You should try to default to julia implemented versions of operations. While the effect is small for `Float64`s it can be catastrophic for `Float32s`, where simd actually provides more accurate results. But `@simd` doesn't always provide more accurate results for arbitrary operations/algorithms. 
"""

# ╔═╡ a58a2b05-c52a-4f1e-a0ff-5266265a44d8
check_test_arr() do 
	column_major(test_arr)
end

# ╔═╡ 343519f1-5b46-4f80-b178-c693318db335
check_test_arr() do 
	row_major(test_arr)
end

# ╔═╡ cc39510b-26f4-4847-a8c8-84b1cfba57cb
check_test_arr() do 
	eachindex_iteration(test_arr)
end

# ╔═╡ af732a86-a239-4943-bb45-bceb4b7c9fd9
check_test_arr() do 
	eachindex_iteration_simd(test_arr)
end

# ╔═╡ 07a50eab-06f0-496d-acb7-7a9cc07cfcdd
check_test_arr() do 
	sum(test_arr)
end

# ╔═╡ 606121d1-76b9-4636-b6e0-1e87aa3bc5a7
md"This effect can be magnified with lower percisions"

# ╔═╡ a7de04d9-6251-4f60-8bcd-7aa4e2d353e2
test_arr_32 = run_example ? rand(Float32, 1000, 100, 32, 4, 4) : nothing;

# ╔═╡ 997e7d85-1fd4-4643-862d-9d448432f795
check_test_arr() do 
	eachindex_iteration(test_arr_32)
end

# ╔═╡ 39bb0893-8f9d-406a-b52a-66e0a5b32531
check_test_arr() do 
	eachindex_iteration(Float64.(test_arr_32))
end

# ╔═╡ 13ae5a80-4486-439a-8b51-0099696b28cb
check_test_arr() do 
	eachindex_iteration_simd(test_arr_32)
end

# ╔═╡ 1737356d-18f4-4e18-a59b-dfcde4f533b6
check_test_arr() do 
	sum(test_arr_32)
end

# ╔═╡ 3359dea1-cad7-41eb-b93a-8de616df43a2
md"""
## Random Numbers

Being considerate about your random number generators is one of the most important aspects of making experiments reproducible (i.e. setting your random seed). Julia lets you set the seed of a Global random number generator, as well as construct and manage your own.
"""

# ╔═╡ 61d9d4d6-30db-47c9-a65f-62907991118a
md"""
There is a global random number generator at `Random.GLOBAL_RNG`, which we can seed using

"""

# ╔═╡ 6e5c8e10-e306-401c-a24a-6d48a8babcc4
Random.seed!(10)

# ╔═╡ 2f931d88-c8aa-47bf-9637-9adaf8fabe5a
md"""
We can generate random numbers via:
"""

# ╔═╡ 64f0fb7a-9b7f-4577-840e-8e24eb4edf1f
rand(2), rand(Float32, 2, 2, 2, 2)

# ╔═╡ 40009e47-22af-40e6-bce0-025b5d267dcc
md"""

Note that we can generate specific types through the call, or just use a default type of `Float64`.

This random number generator is thread local, so when a new thread is created and uses the global rng each thread's global rng will be independent (as of `1.5.x` I believe).

We can also use our own managed RNG:

"""

# ╔═╡ 66da8ba9-08e7-4be0-b3de-dc19114a40a4
rng = Random.MersenneTwister(10)

# ╔═╡ f4e8e1c5-f99d-4ef9-ad3b-41f2a0ba653b
rand(rng, 2), rand(rng, Float32, 2, 2, 2, 2)

# ╔═╡ 0d11335c-d317-4f15-a67c-6285f8a61cc7
md"""
### WARNING

Random number generators are not guaranteed to be consistent across julia versions, and there was a breaking change between 1.4 and 1.5. Because of this there is a package [StableRNGs.jl](https://github.com/JuliaRandom/StableRNGs.jl), which provides stable rngs if needed. This is useful for testing, but if you can guarantee you are going to use a version you should just use RNGs.

"""

# ╔═╡ 01fda67b-3104-475d-baa9-39f291f9d58a
md"""
## Multiple Dispatch

Multiple dispatch is the central design ideology of Julia (much like OOP is central to Python or Java). At first glance, it seems very similar to function overloading of other languages (i.e. C++), but it has much more utility because of the ability to dispatch on all argument types (not just one or two)!  This will be useful later, for now I am only going to simple show how you can take advantage.
"""

# ╔═╡ c85958df-25c2-4fef-b22f-49e76b74cd44
f(x) = "default"

# ╔═╡ 857d7599-2246-4720-93db-f30867be4e42
f(x::Integer) = "Int"

# ╔═╡ 0fa37d34-fa05-42e2-97fb-a4e2d17d0a01
f(x::AbstractFloat) = "Float"

# ╔═╡ a930f429-a89c-446b-9c33-f4374bfa3f41
f("Hello")

# ╔═╡ 2f035f65-5a44-4622-93be-bb682edf65a8
f(1)

# ╔═╡ 1fa7997e-c0f9-46ff-be45-ac4bb205e2ae
f(200f0) # 200f0 is a single precision floating point number

# ╔═╡ 25244e40-c3ba-4146-bb89-2ccd85b04540
md"""
A method is Julia's term for a specialized version of a function. Above we wrote a function `f` and hand-made specialized methods for integers and floats. While this may seem like the compiler is only working on the specialized versions, this is incorrect! The compiler will create a specialized method automatically from the generic function, meaning you get the performance of a hand-specialized method. The overriden methods are useful for when there are code changes for different types (which we'll see later on).

If you specialize a function w/o a generic fallback version you will get an exception that there is no matching method.
"""

# ╔═╡ 08fc585c-27ee-4944-b08d-594e5b28e9e3
function greet(s::String)
    "Hello $(s)"
end

# ╔═╡ effcbd70-c7f9-4602-945d-b1088dbc8ac3
greet("Matthew")

# ╔═╡ 1a59f962-ee18-4325-8462-273f20e0d578
greet(1) # This should throw an exception! greet is not defined for integers!

# ╔═╡ 7cc40fa6-6994-4a27-b882-c5ea079b19e2
md"""
Later in the series you will see how to take advantage of multiple dispatch to design an RL interface and use it to make design easier with composition.
"""

# ╔═╡ 4afda8c7-ae64-4e33-a2f6-535e243318d7
md"""
## Types and Data

Now that we have some of the fundamental building blocks of what makes julia tick, we can start thinking about custom types. First, lets just build a basic struct which contains some data we can act on. As a simple example, lets make a struct A which stores an integer (you can imagine this struct being an agent, environment, or really anything), with a simple function.

Note: The struct and functions are in a `begin..end` block. This has to do with how Pluto works, but not a standard pattern in Julia. 

```julia
struct A
   	data::Int
end

function double(a::A)
    a.data * 2
end
```

This just returns double the data stored in A. Lets make another struct B which holds a string this time

```julia
struct B
   data::String
end
```

we can dispatch on `double` by specializing:

```julia
function double(b::B)
    ret = tryparse(Int, b.data)
    if ret == nothing
        0
    else
        2*ret
    end
end
```

This parses the data in b as an Int and doubles. If it is unable to parse (i.e. the data isn’t an Int) it returns 0.

Great!

Now we can use this in a more complex, but general function

```julia
function complicated_function(a_or_b, args...)
    # ... Stuff goes here ...
    data_doubled = double(a_or_b)
    # other stuff
end
```

Notice how I didn’t specialize the a_or_b parameter above and instead kept it generic. This means any struct which specializes double will slot in the correct function when complicated_function is compiled!

Now it should be pretty clear how you can use multiple dispatch to get the kind of generics you are wanting (even though these are contrived examples). We can abstract one more layer and make this even more usable using abstracts:
"""

# ╔═╡ aa0bc73a-09bb-429c-8038-fbc49aa4a8f8
# create an abstract type, this is the 
# root node of the type hierarchy.
abstract type AbstractAB end

# ╔═╡ df1b192b-d24a-4dd1-a2be-595f40b32544
md"""
Any sub-type of this parent will have 
`my_func` methods defined.

Notice: that these depend on that sub-type
defining `data_as_int` and there is no default 
method. Currently, julia does not have a way to
enforce interfaces like this. See [**Digging Deeper**](#digging-deeper)
for more details.

"""

# ╔═╡ da9710cc-a280-4cab-9c9f-b280eda4fd45
md"Define type `A`"

# ╔═╡ ec8040bf-6872-4cc4-83e0-c886e653542b
begin
	# create a sub-type of AbstractAB
	struct A <: AbstractAB
   		data::Int
	end

	# define its `data_as_int` method to conform to 
	# what is expected by AbstractAB
	data_as_int(a::A) = a.data
end

# ╔═╡ 63c16000-c44b-4a3e-9de7-ab2ec966bc28
md"Define type `B`"

# ╔═╡ 53b6682e-193b-4d25-9cdd-fba523b4b2a9
begin
	# create a sub-type of AbstractAB
	struct B <: AbstractAB
   		data_str::String
	end

	function data_as_int(b::B) 
  		ret = tryparse(Int, b.data_str)
   		if ret == nothing
      		0
   		else
      		ret
   		end
	end
end

# ╔═╡ 02661572-d8b6-40b1-ab2f-eb36ac76405f
md"""
Notice that we moved the complex and specialized code into more restrictive functions so the general functions can be reused. While here we used actual Abstract typing to make dispatch work the way we want, you can also build this exact same interface using duck typing!

**Note**: We can only inherit from abstract types in Julia, so in the above
```julia
struct SubA <: A
	# Stuff
end
```
is not valid. While this can be a little bit tiresome and a bit odd for new users, it adds clarity to the data each struct holds without needing to trace the type hierarchy.
"""

# ╔═╡ ebe712a9-e069-4e3a-950b-b22d8fa08cd3
html"""<h3 id="digging-deeper"> Digging deeper </h3>"""

# ╔═╡ a3d2ef65-63cc-4ae1-8bfb-2b785c3c8255
md"""
Below is another sub-type of `AbstractAB` which doesn't follow the interface guidelines of AbstractAB. Instead there is a bug where `data_as_int` returns `Float64`.
"""

# ╔═╡ dd7fc0c5-70d8-4876-a28a-2023e084a5c0
begin
	# create a sub-type of AbstractAB
	struct C <: AbstractAB
   		data::Float64
	end
	
	# Notice: this will return a float.
	data_as_int(c::C) = c.data
end

# ╔═╡ 7d7d4cb1-61b5-4e1c-af56-fcfcd3b693e1
my_func(aab::AbstractAB) = 20*(data_as_int(aab)^2 + 10)

# ╔═╡ fe98d911-abbe-4194-bde2-c0589efdd864
my_func(A(12)), my_func(B("12"))

# ╔═╡ 934e3e27-5c4f-4aac-8944-0d5821496bca
md"""
Here we define two new functions, which are variants of the above `my_func`. These functions are special in that they guarantee the output of the function is an integer, but with different strategies.

- `return_cast`: will cast the return type of the equation to an integer. 
- `recieve_assert`: will assert that what is returned by data_as_int is an Integer.
"""

# ╔═╡ 207f7e1a-7b18-4b18-b1f3-92198ab05289
return_cast(aab::AbstractAB)::Int = 20*(data_as_int(aab)^2 + 10)

# ╔═╡ 1c36160d-4726-4b2e-9e43-53945b1b7853
recieve_assert(aab::AbstractAB) = 20*(data_as_int(aab)::Int^2 + 10)

# ╔═╡ ba9d77e2-735b-4aef-8f06-56e245b3739c
md"Because the float `3080.0` can be cast to an integer we do this without any loss of information."

# ╔═╡ f7d6d38d-bf14-4d40-9135-f724fa394b58
return_cast.([A(12), B("12"), C(12.0)])

# ╔═╡ 87076bc4-6f59-4cc4-a77f-cd2548988bfb
md"Unlike above `3128.2` can't be exactly cast, so we fail to return the correct value."

# ╔═╡ 9ccf2ef8-1d14-4fb2-84c5-d43aef8087b0
return_cast(C(12.1))

# ╔═╡ 06ced2cc-a530-4b73-9a50-14122b8ebe72
md"Instead of failing based on runtime information, we can assert that the `data_as_int` method for the type of `aab` must return an integer."

# ╔═╡ 43b99729-3859-4ada-ac28-648ae4856550
recieve_assert(C(12.0))

# ╔═╡ 2103a71a-5f21-4561-9f16-28bc0465a988
md"""
We can dig further into the details of the compiled functions using `@code_typed`. Notice how `return_cast` will always do all the operations and then try to cast the float to an int. `recieve_assert` on the other had knows that data_as_int doesn't return a type `Int` for type `C` so will always return a type assertion.
"""

# ╔═╡ 4ee40f79-8f66-4374-8ff2-b516e150e5d4
let
	@code_typed return_cast(C(12.1))
end

# ╔═╡ 6055fb21-bd22-47c9-b69d-32978ae10365
let
	@code_typed recieve_assert(C(12.1))
end

# ╔═╡ 2c725eec-62a7-4d2b-9fb2-241d327a379f
y = 11

# ╔═╡ 21f1a49d-3b15-4eb0-9984-1d19138312c5
y = 12

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.2.2"
PlutoUI = "~0.7.32"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0b5cfbb704034b5b4c1869e36634438a047df065"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "ae6145ca68947569058866e443df69587acc1806"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.32"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─7bd3f2ea-efc3-46d1-a53f-de0e7827459d
# ╟─0216ddd0-c95a-11eb-202a-b5ac98989c51
# ╟─3221dfc7-7ddc-40d1-9ec8-e6d2f61feddf
# ╟─6b7d1012-e2d1-48a2-9e75-eaaf7ebc2a72
# ╟─937863e8-7305-4639-a3dc-3924621ac5d3
# ╟─2fe2603e-8404-4cb0-a9c8-ed7497a88db2
# ╟─15f526ae-8415-470c-aaf3-afb5603428db
# ╟─8df4cb0f-cbdb-452f-a504-4e577e26999d
# ╟─0d84dfe5-ebb8-43fd-8164-2c2c9c06db4b
# ╟─c0c73d7e-488d-440e-88ae-434f923efa2d
# ╟─96dcdb5a-7733-411a-94e1-fc3a9d22fcf3
# ╟─8aeacbc2-252e-4428-9884-be6ef481bfca
# ╠═0c146b61-4322-4d1b-8080-36a9c7b430a9
# ╠═5983b16e-34d7-4b44-9236-ab9b7fa77d5f
# ╟─a62e4f22-6b68-40dd-96cb-8ccc970eeb3b
# ╠═2c725eec-62a7-4d2b-9fb2-241d327a379f
# ╠═21f1a49d-3b15-4eb0-9984-1d19138312c5
# ╟─6e65025b-e625-4445-8dca-99e782d6c2e2
# ╟─75cb3f61-859e-428b-96f6-1c358287baac
# ╠═08aa01ed-eeb5-4296-a4b2-027582c7eeb8
# ╠═f5f4e8dc-d770-477a-94d7-a5468d263d5a
# ╟─04daba2b-17e2-4468-9a56-822e6358ea74
# ╠═5ca1c7ef-dc60-474f-8a2c-03f8ae6143df
# ╠═92af2fb1-a271-4410-8630-41db7559016a
# ╟─40046c3a-2049-46e0-89a0-4d86a5d9345f
# ╟─344ce300-1964-41c1-b19e-5c18b00d12f5
# ╠═14d445fd-b7be-4153-a550-4b486e0b680e
# ╠═da49090c-50ef-4fe4-9eaa-005c2d65e002
# ╟─da024556-51dd-4032-9a70-c346bb442e08
# ╠═4313df31-46ab-4d07-ac22-2970e25e44cf
# ╠═10a0bd50-eae0-40ef-93e1-fa6566a53431
# ╠═baf0edb4-81bc-439d-bdcd-e5f6b9c135fe
# ╟─2d0303f1-985d-440e-9fc4-8ef1e711f47f
# ╠═4cf0e24d-17a7-4d02-990f-2737c9fad982
# ╟─a87690b7-3486-4089-a5ad-91cfe9b34181
# ╠═cfc62088-fb69-429d-aa26-56dbe51430c5
# ╠═566c53c5-c411-45b0-8fe8-c8beff4b372d
# ╟─394ed1d5-b0e4-40ce-9528-1d46d431a786
# ╠═84eadf22-a146-43f9-bb92-9da84059e11a
# ╠═186004b2-3bfb-4fc9-9da5-d39b155542ab
# ╟─d1605fa3-4b02-435e-afb6-1f81dcb4f378
# ╠═244cfc1b-707b-4fd7-b435-2cc3ee5075dc
# ╟─7fee83f9-a41c-47d9-b888-34e0f9bb6e59
# ╠═6c1a3bbe-b465-4088-b76a-bbdd854a58f0
# ╠═70d30b70-4adf-4fcf-bb78-6e731fd14012
# ╟─14e2c8c0-82ea-4c14-b833-c4466d1b7d5d
# ╟─8c0584c5-7754-4a0e-8e75-2ec6f7a42a1b
# ╟─3314a10a-bba6-4552-9de5-c7a843325f7b
# ╠═548f9daa-434a-4786-99bd-48ce5ea782e4
# ╠═08e36c0f-39c7-402e-9e9b-8582b7e0965e
# ╠═d8533b95-764a-44fc-bf79-6ee16a0ca8dc
# ╠═876fe161-7856-434a-993f-adc0ce6e6ccb
# ╠═6fe86984-ca31-4dac-afb8-1dbab8cfb080
# ╟─82b85f5c-a69a-4abe-98a4-8ca9ec16b2c1
# ╠═bc411307-1e1b-41a8-8639-03e0adadb5e6
# ╟─6fb6b701-4c9a-4dc7-80c3-44bd600e3a26
# ╠═a2512e72-6c53-4503-84cd-58f50f34d6e9
# ╠═0ac59e50-84d6-4e29-859a-a94d47cf42c2
# ╟─960aca6b-90e9-4249-aefe-9f4bbce64cc6
# ╠═23b70d5b-5268-4f12-b12b-e3940e3d08fc
# ╠═e0fc96d6-5b27-4baa-8d0b-663e67c3617a
# ╟─486b3b19-af29-42c2-aebb-43b360e0a66f
# ╠═90f24d6a-b842-4cb1-be6e-66b78faa57fc
# ╟─31b53c35-91c3-4c4e-86e0-a48bbd7aa8bf
# ╠═b2524a17-ef89-4ffe-8ab9-bb5034dd1803
# ╠═4b038409-c4ab-4d76-b9ee-90880917ddf5
# ╠═2263d894-745a-46a2-a74f-de82231c9580
# ╟─d98759f8-2e5b-416e-b1d9-da2fc932929a
# ╟─e89e2dde-db5d-43c0-a627-3761e3439b0b
# ╠═0cd3e086-2694-4ed5-a95e-82cc20c64b06
# ╠═0268ec5c-191f-4389-9d07-0da8b213be6b
# ╠═442f3ff7-fcdd-4329-98bc-da10a6c4d8eb
# ╟─4e5567a1-8fcb-49d4-baab-e3d1e21959fc
# ╠═6c654de3-71e2-4263-ac1e-2b9efd477457
# ╟─dc2362c0-f1db-45f0-97aa-2bc5e859c88e
# ╠═5e99e46c-55bd-441a-885a-33134e45ea87
# ╠═95c06ad7-37ae-4765-8a18-414525d07421
# ╟─e681c1a6-c839-4ca1-b2fd-8277b358a47d
# ╠═766ad54e-2e33-4cca-a0b0-12a1ebcbb12e
# ╠═50228280-277a-46cf-a2d9-d71a10196944
# ╠═7f39f7c7-e98b-4054-ba3a-24e3c66d0cf7
# ╟─fa277da9-5355-4a96-929c-8bffb8b98e38
# ╠═becf0487-51c6-47c0-aa4c-8e1578a879bd
# ╟─0edfee8c-d35c-499c-95a2-67b4462fa087
# ╠═09dbc568-bfe1-49ea-b207-e983b7f80359
# ╠═7103f6e0-ecad-4455-87e9-df8ef0eb7399
# ╠═fab829c5-7421-4bc5-9d75-2ae512148e61
# ╟─dce3b77b-089b-4f6a-8f36-4d69af9d874e
# ╠═5fdb2101-6cb8-44a5-b51d-f82bc39cf2ba
# ╟─4b23db15-f28b-477e-8279-bfee0ba30c9f
# ╠═fef36c0d-4ace-4fe8-be21-4d8b6e05119a
# ╠═f079db3f-582b-4df9-af7c-1baa8c6f79b8
# ╠═6f22651a-0c5c-48cb-b5f7-053c70d603b2
# ╠═db737d1c-7a1a-4206-8512-492a0dec0ae6
# ╟─98919df9-476b-4319-a296-ddd39455636a
# ╠═58f6b213-066d-4abe-8ac9-ceb54e340d9b
# ╠═6e4f163b-cc0d-4e06-bcc9-a61ae58ee3e5
# ╠═509599bf-d60e-49e6-853a-dfd950684d4d
# ╠═40ebc8e0-b04d-4216-8e76-50284da2e988
# ╟─15217efb-26ec-43e8-b136-f089084bffa4
# ╠═c86f289f-3a93-4299-863d-7cd2d2235489
# ╟─9a097abc-b457-4a21-8073-bfa312eb1d68
# ╠═f5de53c4-8a33-40f3-9743-07d48830e3bb
# ╟─851249e9-3d96-4c6c-98c6-0bc96d29ae3d
# ╠═87a7a91d-553c-4e00-88eb-bfbb7da1a5ba
# ╟─505a7663-1e7e-4038-93c7-71c91ef6b2ab
# ╠═112055b7-4776-4164-892d-685b098f101a
# ╟─0871989c-298d-458a-a220-7f597a1a07f4
# ╠═2fc9a30b-64bd-4980-910f-d8b8e5083232
# ╟─35b467e3-6bf5-4498-81ab-86376fd15101
# ╠═c4d7efd8-f99e-48f8-9892-38946eb89a17
# ╟─8c3e53b8-037c-4f2a-9cc2-e39358216aa2
# ╠═0e5f5e35-53bb-4d17-bd2b-34bf8fab712d
# ╟─600ad0bd-0eb6-419a-a19f-d0c701910968
# ╠═6c6bec44-ed8e-499b-b1f4-c8753d07698e
# ╟─48a4c4b9-e32f-4fbb-9e57-fe456973032c
# ╠═783ae651-f8c4-4eca-81e2-57c6a807f46a
# ╠═1d595759-9a4f-46f1-b030-3ab96df94fb0
# ╠═30d68c48-1683-4da2-98f5-e910038f0a45
# ╟─7352b072-5933-455c-b757-b8994b4dbe34
# ╠═140391fb-60d1-472a-816f-c9a5800922b8
# ╠═a95d12ac-fe5b-4476-a572-4f5a44d6e9c5
# ╟─331cc1ce-6d52-4919-8e2c-674847fc3f49
# ╟─5c081eb9-9989-4728-b3b6-f9577169d6c8
# ╟─b71dc229-42f3-457c-a8fb-f477d4b2e06c
# ╠═5ced5c9e-899f-44e9-b66e-90bd5348309b
# ╟─611bef16-1eac-4fc4-844b-ef9e7ee23b4c
# ╠═992d5019-a326-4492-9130-2911ed615a5a
# ╠═21960ac5-6816-4aca-b133-35d0406cfa2c
# ╠═a4f25563-815b-44e6-a604-77ea1b55bccb
# ╟─106e8d97-5f74-4fc0-971d-8c3f1b0afeb1
# ╠═f23b08b9-a1ee-48fd-a0b4-96460a30b1b0
# ╟─f0f8c60f-4a07-4edb-9b3d-c063e3190442
# ╠═384d1dff-6f46-456d-839b-b889751e6f03
# ╠═38477409-9d22-4ecc-9094-19eebf38afbf
# ╟─4d70beee-a91f-4ecb-a890-75b1db7bfae0
# ╟─ce200b5c-64b8-468d-b25a-fc630699e2c1
# ╟─35dfe45d-d93e-428d-9fcd-36865a45468f
# ╟─a497e722-1291-479c-98cb-69dd93a6ac9c
# ╟─179ebad2-0804-4b3b-a361-df84903d9bab
# ╟─7d81633e-163e-40d2-a119-8cb2e507be4c
# ╟─4e1fbc51-7df4-4dab-8f9c-ad52cabea74a
# ╠═d4e2cb43-c0b9-4cd3-8dea-3973a20d9da8
# ╟─469fc6da-ab5e-4886-acea-2d94c6139494
# ╠═248d748b-8ce5-401d-94eb-214f440d4fc8
# ╟─0b21b833-f242-4fec-9c09-83d4b9d8628f
# ╟─08ba814a-7a92-4be2-ae15-b94c62b60c2d
# ╠═a58a2b05-c52a-4f1e-a0ff-5266265a44d8
# ╠═343519f1-5b46-4f80-b178-c693318db335
# ╠═cc39510b-26f4-4847-a8c8-84b1cfba57cb
# ╠═af732a86-a239-4943-bb45-bceb4b7c9fd9
# ╠═07a50eab-06f0-496d-acb7-7a9cc07cfcdd
# ╟─606121d1-76b9-4636-b6e0-1e87aa3bc5a7
# ╠═a7de04d9-6251-4f60-8bcd-7aa4e2d353e2
# ╠═997e7d85-1fd4-4643-862d-9d448432f795
# ╠═39bb0893-8f9d-406a-b52a-66e0a5b32531
# ╠═13ae5a80-4486-439a-8b51-0099696b28cb
# ╠═1737356d-18f4-4e18-a59b-dfcde4f533b6
# ╟─3359dea1-cad7-41eb-b93a-8de616df43a2
# ╠═9b54663f-3886-43aa-a6da-9cf84aeedb4d
# ╟─61d9d4d6-30db-47c9-a65f-62907991118a
# ╠═6e5c8e10-e306-401c-a24a-6d48a8babcc4
# ╟─2f931d88-c8aa-47bf-9637-9adaf8fabe5a
# ╠═64f0fb7a-9b7f-4577-840e-8e24eb4edf1f
# ╟─40009e47-22af-40e6-bce0-025b5d267dcc
# ╠═66da8ba9-08e7-4be0-b3de-dc19114a40a4
# ╠═f4e8e1c5-f99d-4ef9-ad3b-41f2a0ba653b
# ╟─0d11335c-d317-4f15-a67c-6285f8a61cc7
# ╟─01fda67b-3104-475d-baa9-39f291f9d58a
# ╠═c85958df-25c2-4fef-b22f-49e76b74cd44
# ╠═857d7599-2246-4720-93db-f30867be4e42
# ╠═0fa37d34-fa05-42e2-97fb-a4e2d17d0a01
# ╠═a930f429-a89c-446b-9c33-f4374bfa3f41
# ╠═2f035f65-5a44-4622-93be-bb682edf65a8
# ╠═1fa7997e-c0f9-46ff-be45-ac4bb205e2ae
# ╟─25244e40-c3ba-4146-bb89-2ccd85b04540
# ╠═08fc585c-27ee-4944-b08d-594e5b28e9e3
# ╠═effcbd70-c7f9-4602-945d-b1088dbc8ac3
# ╠═1a59f962-ee18-4325-8462-273f20e0d578
# ╟─7cc40fa6-6994-4a27-b882-c5ea079b19e2
# ╟─4afda8c7-ae64-4e33-a2f6-535e243318d7
# ╠═aa0bc73a-09bb-429c-8038-fbc49aa4a8f8
# ╟─df1b192b-d24a-4dd1-a2be-595f40b32544
# ╠═7d7d4cb1-61b5-4e1c-af56-fcfcd3b693e1
# ╟─da9710cc-a280-4cab-9c9f-b280eda4fd45
# ╠═ec8040bf-6872-4cc4-83e0-c886e653542b
# ╟─63c16000-c44b-4a3e-9de7-ab2ec966bc28
# ╠═53b6682e-193b-4d25-9cdd-fba523b4b2a9
# ╠═fe98d911-abbe-4194-bde2-c0589efdd864
# ╟─02661572-d8b6-40b1-ab2f-eb36ac76405f
# ╟─ebe712a9-e069-4e3a-950b-b22d8fa08cd3
# ╟─a3d2ef65-63cc-4ae1-8bfb-2b785c3c8255
# ╠═dd7fc0c5-70d8-4876-a28a-2023e084a5c0
# ╟─934e3e27-5c4f-4aac-8944-0d5821496bca
# ╠═207f7e1a-7b18-4b18-b1f3-92198ab05289
# ╠═1c36160d-4726-4b2e-9e43-53945b1b7853
# ╟─ba9d77e2-735b-4aef-8f06-56e245b3739c
# ╠═f7d6d38d-bf14-4d40-9135-f724fa394b58
# ╟─87076bc4-6f59-4cc4-a77f-cd2548988bfb
# ╠═9ccf2ef8-1d14-4fb2-84c5-d43aef8087b0
# ╟─06ced2cc-a530-4b73-9a50-14122b8ebe72
# ╠═43b99729-3859-4ada-ac28-648ae4856550
# ╟─2103a71a-5f21-4561-9f16-28bc0465a988
# ╠═4ee40f79-8f66-4374-8ff2-b516e150e5d4
# ╠═6055fb21-bd22-47c9-b69d-32978ae10365
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
