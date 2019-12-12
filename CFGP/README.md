We can only do this in a very hacky way.

If you want CGCNN to spit out the penultimate layer, then you need to change `cgcnn/cgcnn/model.py`'s line 173 from `return out` to `return out, crys_fea`.
Note that if you do this, then you can no longer train CGCNN.
So you cannot both train CGCNN and predict its penultimate layer using the same exact instance of a CGCNN network object unless you either restart the kernel and reload parameters, you you can get fancy with Jupyter's [`%autoreload`](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) feature.

No, this is not ideal.
We'll fix it if we end up putting this pipeline into production.
