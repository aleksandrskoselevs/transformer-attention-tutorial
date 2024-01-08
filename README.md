Hi, I'm Aleksandrs Koselevs.

- This is a fork of a great pytorch implementation at [sainathadapa/attention-primer-pytorch](https://github.com/sainathadapa/attention-primer-pytorch)
- Which itself is a fork of a great tutorial at [greentfrapp/attention-primer](https://github.com/greentfrapp/attention-primer)
- `attention_primer.ipynb` tries to unify the 5 lessons in a single notebook
- My notes appear as _A: italic_
- All training code is removed in the notebook, only doing inference
- You don't need to train anything. However, there might be some references to `training` or `--parameter=False` in the descriptions
- If you want to train something, the task folders contain the training code
- The implementation in the notebooks might diverge from those in the task folders
- Most images from [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)
- The example code in the descriptions in the notebook might diverge from the one
in _Implementation_
- `batch_size` defaults to 1, to simplify things
- The model implementations are in `all_models.py`, which are copied into the notebooks

## References

[Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.](https://arxiv.org/abs/1706.03762)
