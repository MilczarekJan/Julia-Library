# Create a simple computation graph: f(x, y) = x * y + x
# Which is equivalent to: f(x, y) = x * (y + 1)
include(forward_differentiation.jl)

x = Variable(2.0, nothing)
y = Variable(3.0, nothing)
mult = ScalarOperator{typeof(*)}([x, y], nothing, nothing)
add = ScalarOperator{typeof(+)}([mult, x], nothing, nothing)

# Get the topological order and compute the forward pass
order = topological_sort(add)
result = forward!(order)

println("Result: ", result)  # Expected: 2.0 * 3.0 + 2.0 = 8.0