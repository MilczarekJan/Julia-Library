
update!(node::GraphNode, gradient) =
if isnothing(node.gradient)
    node.gradient = gradient
else
 node.gradient .+= gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    for node in reverse(order)
    backward!(node)
    end
end

function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node,
    [input.output for input in inputs]...,
    node.gradient)
    for (input, gradient) in zip(inputs, gradients)
    update!(input, gradient)
    end
end
   