# Test the forward autodiff implementation
include("forward_differentiation.jl")

# Create a simple computation graph: f(x, y) = x * y + x
# Which is equivalent to: f(x, y) = x * (y + 1)
x = Variable(2.0, nothing)
y = Variable(3.0, nothing)
mult = x * y      # Use the overloaded * operator
add = mult + x    # Use the overloaded + operator

# Alternative explicit construction
# mult = ScalarOperator{typeof(*)}([x, y], nothing, nothing)
# add = ScalarOperator{typeof(+)}([mult, x], nothing, nothing)

# Get the topological order and compute the forward pass
order = topological_sort(add)
result = forward!(order)

println("Result: ", result)  # Expected: 2.0 * 3.0 + 2.0 = 8.0

# Let's also try a slightly more complex example
a = Variable(3.0, nothing)
b = Variable(4.0, nothing)
c = Variable(2.0, nothing)

# Compute: (a * b) / (b + c)
mul_ab = a * b           # 3.0 * 4.0 = 12.0
add_bc = b + c           # 4.0 + 2.0 = 6.0
div_result = mul_ab / add_bc  # 12.0 / 6.0 = 2.0

order2 = topological_sort(div_result)
result2 = forward!(order2)

println("Complex example result: ", result2)  # Expected: 2.0