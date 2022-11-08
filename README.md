# nanograd
Re-implementation of [micrograd](https://github.com/karpathy/micrograd) by Andrej karpathy for learning purposes. Pytorch api like autograd and neural net library. This is a non-vectorized implementation with little to no performance optimizations. 

## Installation

For install,
```
python setup.py install
```

For uninstall,
```
pip uninstall nanograd
```

## Running Examples

**To run examples any of the following methods should work,**
- Install nanograd via `setup.py` and run directly.
- In pycharm right click on parent nanograd folder and mark it as `Sources Root`. IDE errors will be solved and examples can be run directly without install.
- Add the following on top of each example file and run the file directly from that folder. Installation not required.
  ```
  import sys
  sys.path.append("../..")
  ```
- Move examples out to parent nanograd folder and run. Installation not required.



## References

- https://github.com/karpathy/micrograd
- https://www.youtube.com/watch?v=VMj-3S1tku0
