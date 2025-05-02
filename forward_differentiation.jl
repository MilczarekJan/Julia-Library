import Base: +, -, *, /, sin

abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable <: GraphNode
    output :: Any
    gradient :: Any
end

mutable struct ScalarOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
    
    # Constructor
    function ScalarOperator{F}(inputs, output, gradient) where F
        new{F}(inputs, output, gradient)
    end
end

# Convenience constructor
function ScalarOperator(op::Function, x::GraphNode, y::GraphNode)
    ScalarOperator{typeof(op)}([x, y], nothing, nothing)
end

mutable struct BroadcastedOperator{F} <: Operator
    inputs :: Any
    output :: Any
    gradient :: Any
end

# Operator overloading
+(x::GraphNode, y::GraphNode) = ScalarOperator(+, x, y)
-(x::GraphNode, y::GraphNode) = ScalarOperator(-, x, y)
*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
/(x::GraphNode, y::GraphNode) = ScalarOperator(/, x, y)

# Forward pass implementations for scalar operations
function forward(node::ScalarOperator{typeof(+)}, x, y)
    return x + y
end

function forward(node::ScalarOperator{typeof(-)}, x, y)
    return x - y
end

function forward(node::ScalarOperator{typeof(*)}, x, y)
    return x * y
end

function forward(node::ScalarOperator{typeof(/)}, x, y)
    return x / y
end

function topological_sort(head::GraphNode)
    visited = Set()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

function visit(node::GraphNode,
    visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
end

function visit(node::Operator,
    visited, order)
    if node ∉ visited
        push!(visited, node)
        for i in node.inputs
            visit(i, visited, order)
        end
        push!(order, node)
    end
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) = node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end