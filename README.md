# ExplainFix: Explainable Spatially Fixed Deep Convolutional Networks

This library provides open source tools described in the corresponding paper.
The aim to make the tools more generally usable.

Reproducibility Note: All code used for the paper is on GitHub in `./bin/`
and `./dw2/`.  This ExplainFix library was created afterwards to make the
tools easily usable in other contexts.

Check the docstrings for the following functions for details:

    ```
    import explainfix

    # pruning based on spatial convolution layers and saliency
    explainfix.channelprune(my_model, pct=90, ...)

    # explanations of spatial filters
    explainfix.explainsteer_layerwise_with_saliency(...)
    explainfix.explainsteer_layerwise_without_saliency(...)

    # useful tools and some example filters
    explainfix.dct_basis_nd(...)  # DCT basis (Type II by default)
    explainfix.ghaar2d(...)
    explainfix.dct_steered_2d(...)
    ```
