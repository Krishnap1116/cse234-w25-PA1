from typing import Any, Dict, List

import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]
    
class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]
    
class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """
    Op to compute sum along specified dimensions.
    
    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        dim = node.attrs['dim']
        keepdim = node.attrs["keepdim"]

        if keepdim:
            return [output_grad]
        else:
            reshape_grad = expand_as_3d(output_grad, node.inputs[0])
            return [reshape_grad]


class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=0), zeros_like(output_grad)]
    
class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        # print('expand_op',input_tensor.shape, target_tensor.shape)
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=(0, 1)), zeros_like(output_grad)]

class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.
        
        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")
            
        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]
        
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
                
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)
            
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)
            
        return [grad]
class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        """TODO: your code here"""
        return input_values[0]/input_values[1]
    

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        """TODO: your code here"""
        input=node.inputs
        # x1=mul(output_grad,input[0])
        # x2=mul(input[1],input[1])
        # denom=div(x1,x2)
        # denom=mul_by_const(denom,-1)
        # nume=div(output_grad,input[1])

        # return [nume,denom]
        
        return[output_grad/input[1],mul_by_const((output_grad*input[0])/(input[1]*input[1]),-1)]

class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        """TODO: your code here"""
        return input_values[0]/node.attrs['constant']



    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        """TODO: your code here"""
        constant=node.attrs['constant']
        return [output_grad/constant]
class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.
        
        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        """TODO: your code here"""
        # print(input_values[0].shape)
        # if node.attrs['dim0']>0 and node.attrs['dim1']>0:
        #     x1,x2=input_values[0]
        #     dim1=node.attrs['dim0']
        #     dim2=node.attrs['dim1']
        #     y1,y2=input_values[0].clone()
        #     temp=y1[dim1]
        #     temp1=y2[dim2]
        #     x1[dim1]=temp1
        #     x2[dim2]=temp
        #     return input_values[0]
        # else:
        #     print(len(input_values[0][0]))
        #     result=[[0 for i in range(len(input_values[0]))]for j in range(len(input_values[0][0]))]
        #     for i in range(len(input_values[0])):
        #         for j in range(len(input_values[0][0])):
        #             result[j][i]=input_values[0][i][j]
        #     print(result)
        #     return torch.tensor(result)
        dim1=node.attrs['dim0']
        dim2=node.attrs['dim1']
        return torch.transpose(input_values[0],dim1,dim2)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        """TODO: your code here"""
        dim0=node.attrs['dim0']
        dim1=node.attrs['dim1']
        output=transpose(output_grad,dim0,dim1)
        return [output]


class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        """TODO: your code here"""
        # x1,x2 = input_values
        # print(x1)
        # print(x2)
        # print(x2[0])
        # result=[]
        # if len(x1.shape)<3:
        #     x1=x1.unsqueeze(0) 
        #     x2=x2.unsqueeze(0) 
        # length=x1.shape[0]
        # k=0
        # while(length>0):
        #     out=[[0 for j in range(len(x2[k][0]))] for i in range(len(x1[k]))]
        #     for i in range(len(x1[k])):
        #         for j in range(len(x2[k][0])):
        #             for x in range(len(x1[k][0])):
        #                 out[i][j]+=x1[k][i][x]* x2[k][x][j]
        #     k+=1
        #     length-=1
        #     result.append(out)
        # if x1.shape[0]==1:
        #     torch.tensor(result).squeeze(0)
        # return torch.tensor(result)

        x1,x2=input_values
        # print(x1.shape,x2.shape)
        return torch.matmul(x1,x2)
        # length=len(x1.shape)
        # if length>2:
        #     temp=x1.shape[0]
        #     rows=x2.shape[2]
        #     columns=x1.shape[1]
        #     result_rows=x1.shape[1]
        #     result_columns=x2.shape[2]
        #     result_iter=x2.shape[1]
        # else:
        #     temp=1
        #     rows=x2.shape[1]
        #     columns=x1.shape[0]
        #     result_rows=x1.shape[0]
        #     result_columns=x2.shape[1]
        #     result_iter=x2.shape[0]
        # result=[]
        # ind=0
        # while temp>0:
        #     if length>2:
        #         x1_new=x1[ind]
        #         x2_new=x2[ind]
        #     else:
        #         x1_new=x1
        #         x2_new=x2
        #     out=[[ 0 for j in range(rows)]for i in range(columns)]
        #     for i in range(result_rows):
        #         for j in range(result_columns):
        #             for k in range(result_iter):
        #                 out[i][j]+=x1_new[i][k]* x2_new[k][j]
        #             print(out[i][j])
        #     temp-=1
        #     ind+=1
        #     if length>2:
        #         result.append(out)
        #         print(result)
        #     else:
        #         result=out 
        # return torch.tensor(result)                 
        

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input."""
        """TODO: your code here"""
        x1,x2=node.inputs

        return [matmul(output_grad, transpose(x2, dim0=-2, dim1=-1)) , matmul(transpose(x1, dim0=-2, dim1=-1), output_grad)]
        


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        """TODO: your code here"""
        # denom=torch.exp(sum(input_values))
        
        # print(input_values[0].shape)
        # print(node.attrs['dim'])
        # x=input_values[0]
        # rows=input_values[0].shape[0]
        # cols=input_values[0].shape[1]
        # total=[]
        # for i in range(rows):
        #     temp=x[i]
        #     # print(temp)
        #     sum1=0
        #     for i in temp:
        #         sum1+= torch.exp(i)
        #     total.append(sum1)
        # # print(total)

        # for i in range(rows):
        #     for j in range(cols):
        #         input_values[0][i][j]=torch.exp(input_values[0][i][j])/total[i]
        # return input_values[0]
   
        # for i in range(len(input_values[0])):
        #     dim=node.attrs['dim']
        #     nume=torch.exp(input_values[0][i])
        #     denom=torch.sum(nume,dim=dim,keepdim=True)
        #     output=nume/denom
        #     input_values[0][i]=output
        # # print(input_values[0])
        # return input_values[0]
        multiplication=input_values[0]
        dim=node.attrs['dim']
        maxi = torch.max(multiplication, dim=dim, keepdim=True).values
        multi = multiplication - maxi
        exp_multi = torch.exp(multi)
        softmax = exp_multi / torch.sum(exp_multi, dim=dim, keepdim=True)
        return softmax
        

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        return [mul(node, sub(output_grad, sum_op(mul(output_grad,node),dim=node.attrs['dim'],keepdim=True)))] 



            


class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        """TODO: your code here"""
        x=input_values[0]
        mean1 = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)

        # Normalize the input
        eps = node.attrs['eps']
        x= (x - mean1) / torch.sqrt(variance + eps)

        return x
        


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial 
        adjoint (gradient) wrt the input x.
        """
        """TODO: your code here"""

        eps=node.attrs['eps']
        dim=tuple(range(-len(node.attrs['normalized_shape']), 0))
        length = add_by_const(zeros_like(node.inputs[0]), float(torch.prod(torch.tensor(node.attrs['normalized_shape']))))  
        mean1=mean(node.inputs[0],dim=dim,keepdim=True)
        standdev1=mean(power(sub(node.inputs[0],mean1),2.0),dim=dim,keepdim=True)
        x = div(sub(node.inputs[0], mean1), sqrt(add_by_const(standdev1, eps)))
        value1=div(ones_like(node.inputs[0]),sqrt(add_by_const(standdev1, eps)))
        value2 = sub(output_grad, mul(div(ones_like(node.inputs[0]), length), sum_op(output_grad, dim=-1, keepdim=True)))
        value3 = mul(div(ones_like(node.inputs[0]), length), mul(x, sum_op(mul(output_grad,x), dim=-1, keepdim=True)))
        return [mul(value1,sub(value2,value3))]
       

        

class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        """TODO: your code here"""
        # print(input_values[0].shape)
        # rows=input_values[0].shape[0]
        # cols=input_values[0].shape[1]
        # for i in range(rows):
        #     for j in range(cols):
        #         input_values[0][i][j]=max(0,input_values[0][i][j])
        # return input_values[0]
        return torch.relu(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        """TODO: your code here"""
        value= greater(node, zeros_like(node))
        return [mul(output_grad, value)]

class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        """TODO: your code here"""
        x=input_values[0]
        # for i in x:
        #     i=torch.sqrt(i)
        return torch.sqrt(x)
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """TODO: your code here"""

class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        """TODO: your code here"""
        x=input_values[0]
        y=node.attrs['exponent']
        # for  i in x:
        #     i=i**y
        return torch.pow(x,y)
    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """TODO: your code here"""

class MeanOp(Op):
    """Op to compute mean along specified dimensions.
    
    Note: This is a reference implementation for MeanOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        """TODO: your code here"""
        dim1=node.attrs['dim']
        keep=node.attrs['keepdim']
        answer= torch.mean(input_values[0],dim1,keepdim=keep)
        return answer

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """TODO: your code here"""

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()

def topological_sort(nodes):
    """Helper function to perform topological sort on nodes.
    
    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort
        
    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    """TODO: your code here"""
    l=[]
    visited={}
    def dfs(node):
        visited[node]=1
        if isinstance(node, Node):
            for i in node.inputs:
                if i not in visited:
                    # print(i)
                    dfs(i)
        l.append(node)
    nodes=nodes if isinstance(nodes, list) else [nodes]
    for i in nodes:
        if i not in visited:
            # print(i)
            dfs(i)
    # print(l[::-1])
    return l
    
    
    

    


class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        """TODO: your code here"""

        # print(self.eval_nodes)
        # print(input_values)
        
        values={}
        order=topological_sort(self.eval_nodes)
        # print(order)
        for i in order:
            if isinstance(i, Node):
                if i not in input_values:
                    results=[]
                    for j in i.inputs:
                        
                        # print(i)
                        # print(i.inputs)
                        # print(j)
                        if j not in input_values :
                            results.append(values[j])
                            # print(results)
                        else:
                            results.append(input_values[j] )
                            # print(results)
                    # print(values)
                    if i.op!=placeholder:
                        values[i] = i.op.compute(i, results)
                else:
                    values[i]=input_values[i]
            else:  # i is a Tensor
                values[i] = i
        # print(values)
        out=[]
        for i in self.eval_nodes:
            out.append(values[i])
        return out







def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    """TODO: your code here"""
    # print(output_node)
    # print(nodes)
    
    output_values = {}
    x1=nodes[0]
    x2=nodes[1]
    # print('out',output_node)
    # Initialize the gradient for the output node as 1
    output_values[output_node] = ones_like(output_node)  # Gradient of output w.r.t itself is 1
    # print(output_values)
    order=topological_sort(output_node)
    order1=order[::-1]
    # print(order1)
    for i in order1:
        if i not in output_values:
            continue
        if i.op!=placeholder:
            grad=output_values[i]
            # print(grad)
            # print('op',i.op)
            input=i.op.gradient(i,grad)
            # print('input',input)
            # print('i inputs',i.inputs)
            for x, ingrad in zip(i.inputs,input):
                # print('x',x)
                # print('ingrad',ingrad)
                if x not in output_values:
                    output_values[x]=ingrad
                    # print(output_values)
                else:
                    output_values[x]=add(output_values[x],ingrad)
                    # print(add(output_values[x],ingrad))
        # for i in nodes:
        #     if i in output_values:
        #         output_values[i] = sum_op(output_values[i], dim=0)
    return [ output_values.get(node,zeros_like(node)) for node in nodes]

