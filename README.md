# ExplainFix: Explainable Spatially Fixed Deep Convolutional Networks

This library provides open source tools described in the corresponding paper.

(Open Access Article on Wiley Journal of Data Mining and Knowledge Recognition)[https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1483]

### Citation:

  ```
    Gaudio, A., Faloutsos, C., Smailagic, A., Costa, P., & Campilho, A. (2022). ExplainFix: Explainable spatially fixed deep networks. Wiley WIREs Data Mining and Knowledge Discovery, e1483. https://doi.org/10.1002/widm.1483
  ```

  ```
    @article{https://doi.org/10.1002/widm.1483,
        author = {Gaudio, Alex and Faloutsos, Christos and Smailagic, Asim and Costa, Pedro and Campilho, Aur{\'e}lio},
        title = {ExplainFix: Explainable spatially fixed deep networks},
        journal = {Wiley {WIREs} Data Mining and Knowledge Discovery},
        volume = {n/a},
        number = {n/a},
        pages = {e1483},
        keywords = {computer vision, deep learning, explainability, fixed-weight networks, medical image analysis, pruning},
        doi = {https://doi.org/10.1002/widm.1483},
        url = {https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1483},
        eprint = {https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1483}
    }
  ```


### Use the code

Check the docstrings for the following functions for details:

  ```
  $  pip install -U git+https://github.com/adgaudio/ExplainFix.git
  ```

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

  ```
  $ ipython
  In [1]: import explainfix

  In [2]: explainfix.dct_basis_nd((3,3)).shape
  Out[2]: (9, 3, 3)

  In [3]: explainfix.explainsteer_layerwise_with_saliency?
      Signature:
      explainfix.explainsteer_layerwise_with_saliency(
          model: torch.nn.modules.module.Module,
          loader: torch.utils.data.dataloader.DataLoader,
          device: str,
          num_minibatches: int = inf,
          grad_cost_fn: Callable[[ForwardRef('YHat'), ForwardRef('Y')], ForwardRef('Scalar')] = <function <lambda> at 0x7f352adacc10>,
      ) -> explainfix.explainsteer.ExplainSteerLayerwiseResult
      Docstring:
      Apply explainsteer with saliency to all spatial conv2d layers of a model.
      This tells which horizontal and vertical components are most useful to the model.

      Args:
          model: a pytorch model or Module containing spatial 2d convolution layers  (T.nn.Conv2d)
          loader: pytorch data loader
          device: pytorch device, like 'cpu' or 'cuda:0'
          num_minibatches: over how many images to compute the gradients.  We
              think if the images are similar, then you don't actually need a large
              number at all.
          grad_cost_fn: a "loss" used to compute saliency.
              `yhat` is model output. `y` is ground truth.
              The default assumes `yhat=model(x)` and `y` are the same shape.
              Probably `lambda yhat, y: yhat.sum()` also works in many cases.

      Example Usage:
          spectra = explainsteer_layerwise_with_saliency(model, loader, device)
          for layer_idx, (e2, e1, e0) in enumerate(spectra):
              print(layer_idx, e0)
          plot_spectrum_ed(spectra.ed)
      File:      ~/s/r/ExplainFix/explainfix/explainsteer.py
      Type:      function
  ```


### Reproduce paper results:

Reproducibility Note: All code used for the paper is on GitHub in `./bin/`
and `./dw2/`.  This ExplainFix library was created afterwards to make the
tools easily usable in other contexts.

- Download the datasets, with directory structure like below:

    CheXpert:  https://stanfordmlgroup.github.io/competitions/chexpert/

    BBBC038v1: https://bbbc.broadinstitute.org/BBBC038

  ```
  ./data/CheXpert-v1.0-small
      ./train/
      ./train.csv
      ./valid/
      ./valid.csv

  ./data/BBBC038v1_microscopy/
      ./stage1_train/
          ... the unzipped contents of stage1_train.zip here.
      ./stage1_test/
          ... the unzipped contents of stage1_test.zip here.
      ./stage2_test_final/
          ... the unzipped contents of stage2_test_final.zip here.
      ./stage1_solution.csv
      ./stage2_solution_final.csv
  ```

- Install python dependendencies

    ```
    pip install -U simplepytorch
    # or:   pip install -U --no-deps simplepytorch

    pip install -U git+https://github.com/adgaudio/ExplainFix.git

    # whatever else is missing
    ```

- Run the main experiments and plots in the paper

  ```
  redis-server  # run in a separate terminal

  ./bin/all_experiments.sh
  ```
  ... Or do things the manual way (manually do the tasks in all_experiments.sh)

  ```
  # or the manual way of running all experiments
  # (each .sh file outputs a list of commands):
  bash
  . ./bin/bash_lib
  ./bin/chexpert_experiments.sh | run_gpus 1
  ./bin/bbbc038v1_experiments.sh | run_gpus 1
  ./bin/zero_weights.sh | run_gpus 1

  # Reproduce plots

  python bin/fig_baselines.py
  python bin/fig_dct_basis.py
  python bin/fig_ghaar_construction.py
  python bin/fig_heatmap_C8_experiments.py
  python bin/fig_table_num_params_bbbc_baselines.sh
  python bin/zero_weights_plots --exp1 
  python bin/zero_weights_plots --exp2
  python bin/zero_weights_plots --exp3
  python bin/zero_weights_plots --exp4
  python bin/zero_weights_plots --exp5
  # --> NOTE about replicating the BBBC experiment plots: In the #
  bbbc038v1_experiments.sh, I used earlier "versions", V=10 # and V=11, for
  appendix figures (BBBC learning rate and the # in-distribution test).  To
  recreate the figures, you'd need make V=10 or V=11 # and then tweak some
  hardcoded commented-out settings in the plotting file:
  fig_bbbc038v1_plots.py
  # --> Those two appendix figures would therefore probably be challenging to
  re-create.
  simplepytorch_plot BE12 --mode 3 --no-plot -c none < ./bin/fig_bbbc038v1_plots.py
  simplepytorch_plot BE11 --mode 3 --no-plot -c none < ./bin/fig_bbbc038v1_plots.py
  ```
