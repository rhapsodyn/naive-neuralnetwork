# Naive Neuralnetwork

Naive neural network to recognize MNIST handwritten images, no dependencies (except `rand` instead of reading from `/dev/random`), ~80% accuracy.

## Aim: 

1. No matrix -> Less abstraction, easier to understand ??
1. Reuse memory when possible -> Better performance ??
1. Parallel -> Better performance ??

## Result:

1. Every other example chooses matrix for REASONS! Operation like `transpose` helps a lot (for getting rid of `i,j,k`). Avoiding matrix affects readability indeed.
1. `Iter` with references in idiomatic Rust; Less `Copy` more `&mut`. But helps a little, since `Instruments` showed that it has more pressure on CPU than on memory.
1. Not helpful either. (Tried `par_iter` in `Neuron::z()`, got slower.) Because there always has to be SEQUENTIAL: forward after forward, backprop after backprop, iteration after iteration. It would be beneficial with more params (much slower `Neuron::z()`), like 70B? IDK
