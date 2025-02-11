from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        
        x,y=input_values
        multiplication=torch.matmul(x,y)
        x=multiplication
        mean1 = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)

        # Normalize the input
        eps = node.attrs['eps']
        x= (x - mean1) / torch.sqrt(variance + eps)

        return x

    

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        x1, x2 = node.inputs
        eps = node.attrs['eps']
        dim = tuple(range(-len(node.attrs['normalized_shape']), 0))
        multi= matmul(x1, x2)
        length = add_by_const(zeros_like(multi), float(torch.prod(torch.tensor(node.attrs['normalized_shape'])))) 
        mean1=mean(multi,dim=dim,keepdim=True)
        standdev1=mean(power(sub(multi,mean1),2.0),dim=dim,keepdim=True)
        x = div(sub(multi, mean1), sqrt(add_by_const(standdev1, eps)))
        value1=div(ones_like(multi),sqrt(add_by_const(standdev1, eps)))
        value2 = sub(output_grad, mul(div(ones_like(multi), length), sum_op(output_grad, dim=-1, keepdim=True)))
        value3 = mul(div(ones_like(multi), length), mul(x, sum_op(mul(output_grad,x), dim=-1, keepdim=True)))
        out=mul(value1,sub(value2,value3))
        m1=matmul(out, transpose(x2, dim0=-2, dim1=-1))
        m2= matmul(transpose(x1, dim0=-2, dim1=-1), out)
        
        return [m1, m2]
      


       



        


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        x,y=input_values
        multiplication=torch.matmul(x,y)
        dim=node.attrs['dim']
        maxi = torch.max(multiplication, dim=dim, keepdim=True).values
        multi = multiplication - maxi
        exp_multi = torch.exp(multi)
        softmax = exp_multi / torch.sum(exp_multi, dim=dim, keepdim=True)
        return softmax
       

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        
        x1, x2 = node.inputs
        dim = node.attrs['dim']
        multi= matmul(x1, x2)
        soft= softmax(multi)
        out=mul(soft, sub(output_grad, sum_op(mul(output_grad,soft),dim=dim,keepdim=True)))
        m1=matmul(out, transpose(x2, dim0=-2, dim1=-1))
        m2= matmul(transpose(x1, dim0=-2, dim1=-1), out)
        
        return [m1,m2]
    



      

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
