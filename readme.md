# Joint-Bayesian

Implementation of [Joint-Bayesian model](http://home.ustc.edu.cn/~chendong/JointBayesian/) for face verification, written in Python.

### Usage

```
A, G = train(feature, labels)
ratio = verify(A, G, feat1, feat2)
# ratio is the log likelihood between two features.
```

### Dependency

* [NumPy](http://www.numpy.org/)

### Reference

* Chen D, Cao X, Wang L, et al. Bayesian face revisited: A joint formulation[C]// European Conference on Computer Vision. Springer Berlin Heidelberg, 2012: 566-579.
* http://blog.csdn.net/cyh_24/article/details/49059475

### Lincece

[MIT](https://opensource.org/licenses/MIT)


