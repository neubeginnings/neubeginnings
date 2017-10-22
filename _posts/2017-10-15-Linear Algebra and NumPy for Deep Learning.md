---
layout: post
comments: true
---


 This is a combined tutorial on Linear Algebra and NumPy for Deep Learning. The idea behind this tutorial is to intersperse the excellent mathematical content already available on Linear Algebra with programmatic examples to help solidify the basic foundations of both. As this tutorial focuses on the mathematical basics required specifically for Deep Learning, it isn’t mathematically exhaustive but is rather a more practical approach to getting started.

### **Setup**

Install NumPy following the instructions [here](https://www.scipy.org/install.html).

### **Notations**

* Scalars are single numbers. They are usually given lower-case variable names and are written in italics.

* Eg. *s* ∈ R.

* A vector is an array of numbers. They are typically given lower-case variable names and are written in bold. An element of a vector (a scalar) is identified by writing its name in an italic typeface with a subscript.

* Sometimes, we need to index a set of elements of a vector. In this case, we define a set S containing the indices and write the set as a subset. The - sign is used to denote the complement of a set.



* Eg. S = {1, 3, 6} and  $$ \boldsymbol{x}_S $$



![image alt text](image_0.png)

* A matrix is a 2-D array of numbers with each number identified by two indices instead of just one. They are usually given upper-case variable names with a bold typeface.

* Eg. $$\boldsymbol{A} $$

* An element of the matrix is referred to by using its name in italics, but *not* in bold. Eg. The upper left-most entry could be referred to using $$ \textit{A}_{1,1} $$ The transpose of a matrix can be thought of as a mirror image across the main diagonal.

* $$f\left( \boldsymbol{A} \right )_{i, j}  $$ gives the element (i, j) of the matrix computed by applying the function *f*  to **_A_**.

![image alt text](image_1.png)

* Tensors are arrays with more than two axes. They are usually represented by a bold typeface.

* Below is Python code to learn how to define and index scalars, vectors and matrices. The "shape" of an ndarray (numpy array) is its shape along each dimension, and is a very useful debugging tool. Note: These only cover the basics, and you can learn about advanced indexing [here](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

```python
import numpy as np# define a scalars = 1assert np.shape(s) == ()# define a column vectorx = np.array([0, 1, 2])assert x.shape == (3, )# ndarray operations are defined element-wise by defaultx_python_list = [0, 1, 2]assert x_python_list * 2 == [0, 1, 2, 0, 1, 2]assert np.array_equal(x*2, [0, 2, 4])s = [0, 2] # set of indices whose values are to be retrievedassert np.array_equal(x[s], [0, 2])"""np.arange can be used to create a sequencenp.reshape can be used to morph an array into a given shape"""# define a 3x3 matrixA = np.arange(0, 9).reshape(3, 3)assert A.shape == (3, 3)assert A[0, 0] == 0assert np.array_equal(A[0], [0, 1, 2]) # get row 0assert np.array_equal(A[:, 0], [0, 3, 6]) # get column 0 """Slicing basics -Slicing uses the : operator whereas indexing does notSlicing follows the convention; start:end:step - slicing three-tuple (i, j, k)i.e. retrieve elements from the start index to end index jumping in steps=stepDefault values: i = 0 j = n if k&gt;0 -1 if k&lt;0 k = 1Note that :: is the same as : and means select all indices along this axis.For each dimension in the array, we can give i:j:k separated by a , Note: -ve index for start or end is interpreted as len(dimension) + index -ve index for step is interpreted as moving from larger to smaller indexA[0] is indexing row 0A[0:] is slicing from row 0A[:, 0] == A[0:3:1, 0] i.e. slice all rows and index 0th columnNote: Modifying a slice modifies the original array"""
```

### **Multiplying Matrices and Vectors**

An easy way to remember matrix multiplication is by looking at the dimensions of the matrices.

$$ \boldsymbol{A}_{m \times n} \boldsymbol{B}_{n \times p} =\boldsymbol{C}_{m \times p} $$

![image alt text](image_2.png)

#### Vector-Vector Products

Given two vectors $$ x, y \in \mathbb{R}^{n} $$

Inner Product (Dot Product)

Outer Product

![image alt text](image_3.png)

#### Matrix-Vector Products

There are two ways to look at Matrix-Vector products. The first way is when the Matrix is on the left and the vector is on the right, and the second is when the Matrix is on the right and the vector is on the left. Consider a matrix $latex \textbf{\textit{A}} \in \mathbb{R}^{m \times n} $ and a vector $latex x \in \mathbb{R}^n $

##### Case 1: $$ y = \boldsymbol{A}x \in \mathbb{R}^m $$

* Let us write A by rows. Then we find that the ith  entry of $latex y $ is equal to the inner product of the ith row of $latex \textbf{\textit{A}}$ and $latex x$, i.e. $latex y_i = a_{i}^Tx$

![image alt text](image_4.png)

* Let us write **_A_** by columns. Then we can see that *y* is a *linear combination* of the columns of **_A_** where the coefficients of the linear combination are given by the entries of *x*.

![image alt text](image_5.png)

##### Case 2: $$ y^T = x^T \boldsymbol{A} $$

* If we express A by columns, the i<sup>th</sup>  entry of $$y^T $$ is equal to the inner product of $$ x $$ and the i<sup>th</sup> column of $$ \boldsymbol{A} $$

![image alt text](image_6.png)

* If we express A by rows, we see that $$ y^T $$ is a *linear combination* of the rows of **_A_** where the coefficients of the linear combination are given by the entries of *x* .

![image alt text](image_7.png)

#### Matrix-Matrix Products

Case 1: Matrix-Matrix multiplication viewed in the terms of Vector-Vector products.

* If we view **_A_** in rows and **_B_** in columns, the (*i, j*)<sup>th</sup> entry of C is equal to the inner product of *i*<sup>th</sup> row of **_A_** and the *j*<sup>th</sup> column of **_B_** .

![image alt text](image_8.png)

* If we view **_A_** in columns and **_B_** in rows, **_C_** is the sum of all outer products formed by the i<sup>th</sup> column of **_A_** and the i<sup>th</sup>  row of **_B_**.

![image alt text](image_9.png)

Case 2: Matrix-Matrix multiplication viewed in the terms of Matrix-Vector products.
* If we represent **_A_** as the matrix, and **_B_**  by columns, the i<sup>th</sup> column of **_C_** is given by Case-2 of Matrix-Vector products, i.e. vector on the right. *c*<sub>i</sub> = **_A_** *b*<sub>i</sub> .

![image alt text](image_10.png)

* If we represent A in rows and B as the matrix, the ith row of **_C_** is given by Case-1 of Matrix-Vector products, i.e. vector on the left. *c*<sub>i</sub><sup>T</sup> = *a*<sub>i</sub><sup>T</sup> **_B_** .

![image alt text](image_11.png)

* Some useful properties of Matrix multiplication

    * It is associative: (**_AB_**)**_C_** = **_A_**(**_BC_**).

    * It is is distributive: **_A_**(**_B_** + **_C_**) = **_AB_** + **_AC_**.

    * It is not commutative; **_AB_** $$ \neq $$ **_BA_**.

### Linear Dependence and Span

For a given system of equations **_Ax_** = **_b_**, it is possible to have exactly one solution, no solution or infinitely many solutions for some values of **_b_**. It is *not* possible, however, to have more than one but less than infinitely many solutions for a particular b. If we assume that  **_x_**  and **_y_** are solutions, then **_z_** = α **_x_** + (1 − α) **_y_** is also a solution for any real α.



The linear combination of a set of vectors is given by multiplying (scaling) each vector with a scalar, and then adding the result. The span of a set of vectors is the set of ALL points obtained by linear combination of the original vectors.

Each column of **_A_** can be looked at as a direction from the origin (vector of zeros), and **_x_** is the amount to travel in that direction. Determining whether **_Ax_**= **_b_** has a solution amounts to testing whether **_b_** is in the span of the columns of **_A_** (known as range or column space of **_A_**.)

If **_b_** is m-dimensional, then there need to be at least m columns in A, to move in m-directions to reach **_b_** from the origin. If this is not the case, there will be some values for **_b_** that can’t be reached using the directions of **_A_**, and the equation might have no solution. However, even if it has m-columns, if two columns are identical in their direction, one of them would be redundant. Formally, this redundancy is known as *linear dependence* . A set of vectors is *linearly independent* if no vector in the set is a linear combination of the others.

To ensure that there’s only one solution for **_b_**, we must ensure that **_A_** has at most m columns, otherwise there is more than one way of parameterizing each solution. Hence, to use the matrix inversion method, **_A_** must be a nonsingular square matrix.

The column/row rank of a matrix **_A_**<sub>m x n</sub> is the largest number of columns/rows respectively of A that constitute a linearly independent set. It turns out that for any matrix, column rank = row rank, and are collectively referred to as the rank of A.

![image alt text](image_12.png)

```python

import numpy as np

from numpy.linalg import matrix_rank

I = np.eye(4)

assert matrix_rank(I) == 4 # Full rank identity matrix

I[-1,-1] = 0. # Create a rank deficient matrix

assert matrix_rank(I) == 3

O = np.ones((4,))

assert matrix_rank(O) == 1 # 1 dimension - rank 1 since all vectors are the same

O = np.zeros((4,))

assert matrix_rank(O) == 0 # Vectors don’t go in any direction

```

### Norms

The size of the vector is usually measured by a function called norm. Formally, the $$ L^{p} $$ is given as :

![image alt text](image_13.png)

Norms are functions mapping vectors to nonnegative values. Intuitively norm measures the distance of the point from the origin. More rigorously, norm is a function which satisfies the following properties

![image alt text](image_14.png)

The $$ L^2 $$ norm is called the Euclidean norm. It is used so frequently that the subscript p is dropped. It is also common to measure which is simple

### Special Kinds of Matrices and Vectors

1. A diagonal matrix has entries only along the main diagonal. In **_D_**, **_D_**<sub>i,j</sub> = 0 if i $$ \neq $$  j. We can write diag( **_v_** ) to denote a square **_D_** whose diagonal entries are given by **_v_** .To compute diag( **_v_** ) **_x_** , we only need to scale each element x<sub>i</sub> by v<sub>i</sub>, i.e. diag( **_v_** )( **_x_** ) = **_v_** $$\odot$$ **_x_** . The inverse of **_D_** exists if every diagonal entry is nonzero, and diag( **_v_** )  -1 = diag ([1/v<sub>1</sub>, … , 1/v<sub>n</sub>]<sup>T</sup> ). We may derive a very general ML algorithm in terms of arbitrary matrices, but obtain a less expensive (and less descriptive) algorithm by restricting some matrices to be diagonal. Non square diagonal matrices can be multiplied cheaply after concatenating zeroes to the result if **_D_** is taller than wide or discarding some of the last elements if **_D_** is wider than tall.

2. A symmetric matrix is one that is equal to its own transpose, i.e. **_A_** = **_A_** <sup>T</sup>. An antisymmetric is one where **_A_** = - **_A_** <sup>T</sup>. **_A_** + **_A_** <sup>T</sup> is symmetric and **_A_** - **_A_** <sup>T</sup> is antisymmetric. Thus, every square matrix can be represented as a sum of a symmetric and an antisymmetric matrix; **_A_** = 0.5 * (**_A_** + **_A_** <sup>T</sup>) + 0.5 * (**_A_** - **_A_** <sup>T</sup>).

3. A unit vector is a vector with unit norm, i.e. $$ \| x \|_2 = 1 $$

4. A vector **_x_** and a vector **_y_** are orthogonal to each other if **_x_** <sup>T</sup> **_y_** = 0. If two vectors are orthogonal and have unit norm, they are orthonormal. An orthogonal matrix is a square matrix whose rows are mutually orthonormal, and whose columns are mutually orthonormal, i.e. **_A_** <sup>T</sup> **_A_** = **_AA_** <sup>T</sup> = **_I_**, implying **_A_** <sup>-1</sup> = **_A_** <sup>T</sup>. Note that operating on a vector with an orthogonal matrix will not change its Euclidean norm.

```python
import numpy as np

import scipy

D = np.diag([1, 2, 4]) # Create a diagonal matrix

D_I = np.linalg.inv(D) # Find inverse of D

# Calculate the inverse based on diagonal property

diagonal = np.diag(D)

assert np.array_equal(D_I, np.diag(1.0 / diagonal))

"""

Create a symmetric matrix.

Fill A[i][j] and autofill a[j][i]

"""

A = np.zeros((3, 3))

A[0][1] = 1

A[1][2] = 2

A = A + A.T - np.diag(A.diagonal())

assert np.all(A == A.T) # Check for symmetric

# Some methods to create an orthogonal matrix

R = np.random.rand(3, 3)

Q = scipy.linalg.orth(R)

Q = scipy.stats.ortho_group.rvs(dim=3)

Q, _ = np.linalg.qr(R)

```

### Eigen decomposition

An eigenvector of a square matrix **_A_** is a non-zero vector **_v_** such that multiplication by **_A_** alters only the scale of **_v_** : **_Av_** = λ **_v_** . Here, the scalar λ is the eigenvalue corresponding to the eigenvector **_v_** (Note: We can also find the left eigenvector such that **_v_** <sup>T</sup> **_A_** = λ **_v_** <sup>T</sup> ) If v is an eigenvector, then so is any rescaled vector sv (if s $$ \in \mathbb{R}, s > 0) $$ with the same eigenvalue.

Suppose that a matrix **_A_** has n linearly independent eigenvectors, {**_v_** <sup>(1)</sup>, . . . ,**_v_** <sup>(n)</sup>}, with corresponding eigenvalues {λ<sub>1</sub>, . . . , λ<sub>n</sub>}. We may concatenate all of the eigenvectors to form a matrix **_V_** with one eigenvector per column. Likewise, we can concatenate the eigenvalues to form a vector **_λ_** .

* The eigendecomposition of **_A_** is then given by **_A_** = **_V_** diag( **_λ_** ) **_V_** <sup>−1</sup>.

* Not every matrix can be decomposed into eigenvalues and eigenvectors. In some cases, the decomposition exists, but may involve complex rather than real numbers.

* Every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and eigenvalues: **_A_** = **_Q_** 𝜦 **_Q_** <sup>T</sup> Where Q is an orthogonal matrix composed of eigenvectors of A, 	𝜦 is a diagonal matrix,  Eigenvalue 𝛬<sub>i,i</sub> is associated with the eigenvector in column i of **_Q_**.  Because **_Q_** is an orthogonal matrix, we can think of **_A_** as scaling space by λ<sub>i</sub> in direction **_v_**<sup>(i)</sup>.

* Eigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue, and we could equivalently choose a **_Q_** using those eigenvectors instead. By convention, we usually sort the entries of **_Λ_** in descending order. Under this convention, the eigendecomposition is unique only if all of the eigenvalues are unique.

* A matrix whose eigenvalues are all positive is called positive deﬁnite. A matrix whose eigenvalues are all positive or zero-valued is called positive semideﬁnite. Likewise, if all eigenvalues are negative, the matrix is negative deﬁnite, and if all eigenvalues are negative or zero-valued, it is negative semideﬁnite. Positive semideﬁnite matrices are interesting because they guarantee that ∀ **_x_** , **_x_** <sup>T</sup> **_Ax_** ≥ 0.Positive deﬁnite matrices additionally guarantee that **_x_** <sup>T</sup> **_Ax_** = 0 ⇒ **_x_** = 0

![image alt text](image_15.png)

![image alt text](image_16.png)

* Note that only the symmetric part of **_A_** contributes to the quadratic form and so we often implicitly assume that the matrices appearing in a quadratic form are symmetric.

* One important property of positive definite and negative definite matrices is that they are always full rank, and hence, invertible.

* Given any matrix **_A_** ∈ R<sub>m×n</sub> (not necessarily symmetric or even square), the matrix **_G_** = **_A_** <sup>T</sup> **_A_** (sometimes called a Gram matrix) is always positive semidefinite. Further, if m ≥ n (and we assume for convenience that A is full rank), then G = **_A_** <sup>T</sup> **_A_** is positive definite.

* An application where eigenvalues and eigenvectors come up frequently is in maximizing some function of a matrix max  **_x_** ∈ R<sub>n</sub> **_x_** <sup>T</sup> **_Ax_** subject to $$ \| x \|_{2}^2  $$ = 1. Assuming the eigenvalues are ordered as λ<sub>1</sub> ≥ λ<sub>2</sub> ≥ . . . ≥ λ<sub>n</sub>, the optimal x for this optimization problem is x<sub>1</sub>, the eigenvector corresponding to λ<sub>1</sub>. We can also solve the minimization problem in the same way.

The following NumPy functions can be used for eigenvalue decomposition.

![image alt text](image_17.png)

Please note that SciPy also has a linalg package. If possible, it is preferable to use the SciPy linalg package (after installing an optimized version of SciPy, because NumPy was built to be portable rather than fast and cannot use a FORTRAN compiler. SciPy on the other hand has access to the FORTRAN compiler and many lower level routines, and hence is preferred.)

### Singular Value Decomposition

The previous section demonstrated decomposition of a matrix into eigenvalues and eigenvectors, however eigen decomposition is not generally applicable as a non-square matrix does not have an eigen decomposition. Another way to decompose a matrix is to factorize into singular values and singular vectors and this is called Singular Value Decomposition and is applicable for any real matrix. SVD is similar, except is defined as:

![image alt text](image_18.png)

Suppose that **_A_** is an m x n matrix. Then **_U_** is  defined to be an m x m matrix, **_D_** an m x n matrix, and **_V_** to be an n x n matrix.

**_U_** : Orthogonal matrix, columns are known as left-singular vectors and are the eigenvectors of **_AA_** <sup>T</sup>

**_V_** : Orthogonal matrix, columns are known as right-singular vectors and are the eigenvectors of **_A_** <sup>T</sup> **_A_**

**_D_** : Diagonal matrix, elements along the diagonal are known as singular values and are the non-zero singular values are the square roots of eigenvectors of **_AA_** <sup>T</sup> or **_A_** <sup>T</sup> **_A_**

SVD is useful as it allows us to partially generalize matrix inversion to non-square matrices.

```python

import numpy as np
"""    np.linalg.svd can be used to find the SVD for A    It factorizes the matrix A as U * np.diag(d) * V
"""
A = np.random.rand(4, 3)  # Create a random matrixU, d, V = np.linalg.svd(A)  # Finds the SVD for matrix AD = np.zeros((4, 3))D[:3, :3] = np.diag(d)  # Slice and assign the diagonalassert U.shape == (4, 4)assert V.shape == (3, 3)assert np.allclose(A, np.dot(U, np.dot(D, V)))  #Reconstruct the matrix and compare

```

### Moore-Penrose Pseudoinverse

Matrix inversion is not defined for non-square matrices. Recall the linear equation

![image alt text](image_19.png)

If **_A_** is taller than it is wide, then it is possible for this equation to have no solution. If **_A_** is wider than it is tall, there could be multiple possible solutions. The pseudoinverse is defined as :

![image alt text](image_20.png)

Practical definitions for computing pseudoinverse are not based on this formula, but are derived from the SVD. Hence, we can define the pseudoinverse **_A_**<sup>+</sup> as :

![image alt text](image_21.png)

Where **_U_**, **_D_**, **_V_** are the singular value decomposition of A. Note that the pseudoinverse of **_D_** is obtained by taking the reciprocal of its non-zero elements and then taking the transpose of the resulting matrix.

If A has more columns than rows (wider than tall), the pseudo inverse provides one of the many solutions, specifically it provides the solution with the minimum Euclidean norm.

$$ x = A^{+}y $$ where x is the solution with the minimal Euclidean norm $$ \| x \|_2 $$


If A has more rows than columns (taller than wide), it is possible for there to be no solution. In this case, the solution x obtained minimizes the Euclidean norm $$ \| Ax - y \|_2 $$

```python

import numpy as npA = np.random.rand(3, 3)B = np.linalg.pinv(A)assert np.allclose(A, np.dot(A, np.dot(B, A))) # BA should be close to an identity matrix

```

### Trace

The trace operator gives the sum of all the diagonal entries of a matrix:

![image alt text](image_22.png)

Trace operator allows us to conveniently specify certain operations. For example, the Frobenius norm can be elegantly defined as :

![image alt text](image_23.png)

Some properties of the trace operator:

* Invariant to transpose :  $$ Tr(\boldsymbol{A}) = Tr(\boldsymbol{A}^{T})  $$

* Scalar has its own trace $$ Tr(a) = a $$

* Invariance to cyclic permutation : $$ Tr(\boldsymbol{A B C}) = Tr(\boldsymbol{C A B}) = Tr(\boldsymbol{B C A}) $$. This holds true even when the shape of the matrices are different. More generally,

### ![image alt text](image_24.png)

```python

import numpy as npfrom math import sqrtA = np.arange(9).reshape(3, 3)assert np.trace(A) == 12norm = np.linalg.norm(A)                     # Compute Frobenius normassert sqrt(np.trace(A.dot(A.T))) == norm    # Compare with the result from trace operator

```

### Determinant

Determinant maps a matrix to a real valued scalar. Determinant is equal to the product of all the eigenvalues of the matrix. It can be thought of how much multiplication by the matrix expands or contracts space. If determinant is 0, the matrix loses all of its volume and is contracted completely across at least one dimension. If determinant is 1, then  the transformation preserves volume. Determinant is denoted as $$ det(\boldsymbol{A}) $$

```python

import numpy as np
A = np.array([[1, 2], [3, 4]])print np.linalg.det(A)

```

### Matrix Calculus

Coming soon!

### **References**

[1] [http://www.deeplearningbook.org/contents/linear_algebra.html](http://www.deeplearningbook.org/contents/linear_algebra.html)

[2] [http://cs229.stanford.edu/section/cs229-linalg.pdf](http://cs229.stanford.edu/section/cs229-linalg.pdf)

[3] [http://cs231n.github.io/python-numpy-tutorial/](http://cs231n.github.io/python-numpy-tutorial/)

[4] [http://www.bogotobogo.com/python/python_numpy_array_tutorial_basic_A.php](http://www.bogotobogo.com/python/python_numpy_array_tutorial_basic_A.php)

[5] [http://www.bogotobogo.com/python/python_numpy_array_tutorial_basic_B.php](http://www.bogotobogo.com/python/python_numpy_array_tutorial_basic_B.php)

[6] [http://www.bogotobogo.com/python/python_numpy_matrix_tutorial.php](http://www.bogotobogo.com/python/python_numpy_matrix_tutorial.php)


{%if page.comments %}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://neubeginnings.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}
