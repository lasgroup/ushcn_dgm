# Distributional Gradient Matching for Learning Uncertain Neural Dynamics Models

[**Requirements**](#requirements)
| [**Training**](#training)
| [**Results**](#results)
| [**Contributing**](#contributing)

This repository contains code for reproducing the USHCN experiments of the PhD thesis of Philippe Wenk.


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> We build our code using [JAX](https://github.com/google/jax). The code of the algorithm is in the folder [`dgm`](./dgm).

## Training

All example training files can be found in the folder [`examples`](./examples/ushcn). 

## Contributing

If you would like to contribute to the project please reach out to [Lenart Treven](mailto:trevenl@ethz.ch?subject=[DGM]%20Contribution%20to%20DGM) or [Philippe Wenk](mailto:philippewenk@hotmail.com?subject=[DGM]%20Contribution%20to%20DGM). If you found this library useful in your research, please consider citing.
```
@article{treven2021distributional,
  title={Distributional Gradient Matching for Learning Uncertain Neural Dynamics Models},
  author={Treven, Lenart and Wenk, Philippe and Dörfler, Florian and Krause, Andreas},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
