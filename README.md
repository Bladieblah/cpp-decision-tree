# cpp-decision-tree
Basic decision tree implementation in C++ wrapped with python, that I wrote for a training I organized.


## Setting up
Make a venv and install the requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Compiling your code

Compile by using the Makefile, you can test your code by modifying the `main` function in [decision_tree.cpp](/src/decision_tree_c/decision_tree.cpp).

```bash
make
```

## Completing the exercises
All the boilerplate code you need is located in the [decision_tree_c](/src/decision_tree_c/) folder. To compile and install the module, run

```bash
make install
```

There is also a simple unittest script, you can run it with 

```bash
make test
```
