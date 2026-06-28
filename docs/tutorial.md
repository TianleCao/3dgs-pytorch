---
title: "Learning notes on 3DGS math and implementations"
description: "Notes about EWA splatting, adpative control, and some less apparent learnings from the official 3DGS repo"
date: "Jun 25 2026"
---
# [Source Code](https://github.com/TianleCao/3dgs-pytorch)

# Learning notes on 3DGS math and implementations

## Big picture

The problem 3DGS solves is **novel view synthesis**: given a small set of photographs of a static scene from known camera positions, build a renderer that can show you the same scene from any other camera position. The hard part is that a few photos don't tell you everything about the 3D structure — you have to *infer* a 3D representation that is consistent with everything you observed.

Different methods make different choices about what that 3D representation looks like:

- **Classical meshes** store geometry as triangles. Fast to render, but hard to recover from photos automatically and not great at "fuzzy" geometry like leaves, hair, or smoke.
- **NeRF** (Mildenhall, 2020) stores the scene as an MLP that, given a 3D position and a view direction, returns the color and density at that point. Rendering means sampling many points along each pixel's ray and integrating. Expressive, but slow at both training and rendering.
- **3DGS** stores the scene as a large collection of **3D Gaussians** — soft, oriented blobs in space. Each blob has a position, a shape (covariance), an opacity, and a view-dependent color. Rendering means projecting the blobs onto the image plane and alpha-compositing them front-to-back.

The 3DGS choice is interesting because it is simultaneously **explicit** (you can list every Gaussian and inspect them) and **soft** (Gaussians overlap and blend, naturally handling subtle or "fuzzy" geometry). The rendering is a closed-form formula, not a network forward pass, which is why 3DGS can render at 100+ FPS where vanilla NeRF rendered at minutes per frame.

The training loop is conceptually simple:

1. **Render** the scene from a known camera using the current set of Gaussians.
2. **Compare** the rendered image to the ground-truth photo from that camera (L1 + D-SSIM loss).
3. **Backpropagate** the pixel-wise error through the differentiable rasterizer to get gradients on every Gaussian parameter.
4. **Update** the parameters (Adam optimizer, with different learning rates per parameter group).
5. **Periodically grow or shrink the Gaussian population** — split overgrown ones, clone undergrown ones, prune dead ones — so the representation adapts to the local complexity of the scene.

The rest of this document goes through each of these steps. Section 0 covers what each Gaussian "is" (its parameters and their activations). Section 1 covers rendering — the math that turns Gaussians into pixels while letting gradients flow back. Section 2 covers adaptive control (the dynamic growth and pruning). Section 3 collects implementation details that surprised me when I compared my implementation to the reference code.

## 0. Parameterization

Different from NeRF which use a MLP to store the scene implicitly, 3DGS models scenes as the summation of many 3D gaussians. Each gaussian are parameterized with:
- position ($3$ params)
- cov ($7$ aprams)
	- rotation (in quaternions, $4$ params)
	- scaling ($3$ params). Note scaling is expressed in the log space, i.e. an `exp()` function will be applied as an activation to make sure if will always be positive
- opacity ($1$ param). Note that opacity also uses an activation function, i.e. `sigmoid()` to make sure it is always within 0 to 1.
- spherical harmonics ($16$ params if using `deg=3`). One can simply understand it some intrinsics which will define the color of gaussian at different viewing angles.
## 1. Rendering

Different from NeRF (ray based rendering), 3DGS uses point-based rendering. The rendering process simply "throws" the gaussians onto the canvas (camera detector), so that one can compare that with the training data and use the differences to guide how to revise the gaussians. 

### 1.1 Coordinate transform

A 3D Gaussian is characterized by its mean $\boldsymbol{\mu}_w$ (a 3-vector) and covariance $\Sigma_w$ (a 3×3 matrix) in **world space**. To render it from a particular camera, we need to know where it lands in **camera space** — the coordinate frame where the camera sits at the origin and looks down its own axis.

**The transform itself.** Let the camera-to-world transform (which is what the dataset typically stores) be a rigid motion $[R \vert t]$: rotate by $R$, then translate by $t$. The inverse — world-to-camera — is $[R^\top \vert -R^\top t]$, so a world point $\mathbf{x}_w$ becomes:

$$\mathbf{x}_c = R^\top \mathbf{x}_w + (-R^\top t) = R^\top(\mathbf{x}_w - t)$$

This is an **affine function** of $\mathbf{x}_w$ — a linear part ($R^\top$) plus a constant offset ($-R^\top t$). Affine functions of Gaussian random variables stay Gaussian, and the new mean and covariance follow from two short derivations.

**How the mean transforms.** Take expectations on both sides:

$$\mathbb{E}[\mathbf{x}_c] = R^\top \mathbb{E}[\mathbf{x}_w] + (-R^\top t) = R^\top(\boldsymbol{\mu}_w - t)$$

So $\boldsymbol{\mu}_c = R^\top(\boldsymbol{\mu}_w - t)$ — the mean transforms exactly like an ordinary point. The translation matters here.

**How the covariance transforms.** Start from the definition:

$$\mathrm{Cov}(\mathbf{y}) = \mathbb{E}[(\mathbf{y} - \mathbb{E}[\mathbf{y}])(\mathbf{y} - \mathbb{E}[\mathbf{y}])^\top]$$

Apply it to $\mathbf{x}_c$:

$$\mathbf{x}_c - \mathbb{E}[\mathbf{x}_c] = R^\top \mathbf{x}_w - R^\top t - R^\top(\boldsymbol{\mu}_w - t) = R^\top(\mathbf{x}_w - \boldsymbol{\mu}_w)$$

Substituting:

$$\Sigma_c = \mathbb{E}[R^\top (\mathbf{x}_w - \boldsymbol{\mu}_w)(\mathbf{x}_w - \boldsymbol{\mu}_w)^\top R] = R^\top \mathbb{E}[(\mathbf{x}_w - \boldsymbol{\mu}_w)(\mathbf{x}_w - \boldsymbol{\mu}_w)^\top] R = R^\top \Sigma_w R$$

So the covariance picks up an $R^\top$ on the left and an $R$ on the right — it is "sandwiched" by the rotation. Two things to notice:

- **Translation $t$ doesn't appear in $\Sigma_c$.** Covariance is translation-invariant by definition: shifting every point in space by the same vector doesn't change how spread out they are. This is why only $R$ shows up.
- **The covariance is sandwiched, not just multiplied once.** The pattern $A \Sigma A^\top$ is general: for any linear $\mathbf{y} = A\mathbf{x}$, $\mathrm{Cov}(\mathbf{y}) = A \mathrm{Cov}(\mathbf{x}) A^\top$. It's easy to get wrong if you think of the Gaussian as just a point and apply $R^\top$ alone.

Putting it together:

$$\mathbf{x}_c \sim \mathcal{N}(R^\top(\boldsymbol{\mu}_w - t), R^\top \Sigma_w R)$$

### 1.2 3D to 2D (EWA splatting)

Once inside the camera frame, we have a 3D Gaussian $\mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c)$ that we want to project onto the image plane.

**Projecting the mean** is straightforward. Apply the standard pinhole projection from camera intrinsics $K$:

$$u = f_x \frac{x_c}{z_c} + c_x, \qquad v = f_y \frac{y_c}{z_c} + c_y$$

**Projecting the covariance** is where things get interesting. The trick from Section 1.1 (rule "linear function of a Gaussian is a Gaussian") only worked because the world-to-camera transform was affine. Perspective projection is **not linear** — it has $1/z_c$ in it, so the image of a 3D Gaussian under projection is *not* a 2D Gaussian in general. We can't reuse the same rule directly. We need an approximation that lets us keep treating splats as Gaussians on the image plane.

**The EWA idea: locally linearize.** Even though the projection

$$\pi(x, y, z) = \left( f_x \frac{x}{z}, f_y \frac{y}{z} \right)$$

is nonlinear globally, it can be approximated by its **first-order Taylor expansion** locally. For a Gaussian centered at $\boldsymbol{\mu}_c = (x_c, y_c, z_c)$ and a small displacement $\boldsymbol{\delta}$:

$$\pi(\boldsymbol{\mu}_c + \boldsymbol{\delta}) \approx \pi(\boldsymbol{\mu}_c) + J \boldsymbol{\delta}$$

where $J$ is the Jacobian of $\pi$ — the $2 \times 3$ matrix of first partial derivatives — evaluated at $\boldsymbol{\mu}_c$. Computing each entry:

$$
\frac{\partial}{\partial x}\left(f_x \frac{x}{z}\right) = \frac{f_x}{z}, \quad \frac{\partial}{\partial y}\left(f_x \frac{x}{z}\right) = 0, \quad \frac{\partial}{\partial z}\left(f_x \frac{x}{z}\right) = -\frac{f_x x}{z^2}
$$

$$
\frac{\partial}{\partial x}\left(f_y \frac{y}{z}\right) = 0, \quad \frac{\partial}{\partial y}\left(f_y \frac{y}{z}\right) = \frac{f_y}{z}, \quad \frac{\partial}{\partial z}\left(f_y \frac{y}{z}\right) = -\frac{f_y y}{z^2}
$$

Assembling, evaluated at $(x_c, y_c, z_c)$:

$$
J = \begin{bmatrix} f_x/z_c & 0 & -f_x x_c / z_c^2 \cr 0 & f_y/z_c & -f_y y_c / z_c^2 \end{bmatrix}
$$

**Closing the loop.** Let $\mathbf{u}_c = \pi(\mathbf{x}_c)$ be the projected 2D position of a sample $\mathbf{x}_c \sim \mathcal{N}(\boldsymbol{\mu}_c, \Sigma_c)$. Within the Taylor approximation:

$$\mathbf{u}_c \approx \pi(\boldsymbol{\mu}_c) + J(\mathbf{x}_c - \boldsymbol{\mu}_c)$$

Taking expectations on both sides — the linear displacement term $J(\mathbf{x}_c - \boldsymbol{\mu}_c)$ has zero mean because its inner factor does — gives the 2D mean as the exact projection of the 3D center:

$$\boldsymbol{\mu}_{2D} = \mathbb{E}[\mathbf{u}_c] \approx \pi(\boldsymbol{\mu}_c)$$

For the covariance, apply the definition $\mathrm{Cov}(\mathbf{u}_c) = \mathbb{E}[(\mathbf{u}_c - \mathbb{E}[\mathbf{u}_c])(\mathbf{u}_c - \mathbb{E}[\mathbf{u}_c])^\top]$. The centered version of $\mathbf{u}_c$ is:

$$\mathbf{u}_c - \mathbb{E}[\mathbf{u}_c] \approx J(\mathbf{x}_c - \boldsymbol{\mu}_c)$$

Substituting:

$$\Sigma_{2D} = \mathbb{E}[J(\mathbf{x}_c - \boldsymbol{\mu}_c)(\mathbf{x}_c - \boldsymbol{\mu}_c)^\top J^\top] = J \mathbb{E}[(\mathbf{x}_c - \boldsymbol{\mu}_c)(\mathbf{x}_c - \boldsymbol{\mu}_c)^\top] J^\top = J \Sigma_c J^\top$$

Same "sandwich" pattern as in Section 1.1 — and for the same reason: the centered displacement passes through the linear part of the Taylor expansion, the expectation pulls out the constant matrices on either side, and what's left in the middle is the original covariance. The mean of the splat is $\pi(\boldsymbol{\mu}_c)$ — the exact projection of the 3D center, since a single point doesn't need any approximation.

This approximation is **exact at $\boldsymbol{\mu}_c$** and increasingly accurate when the Gaussian's spread in 3D is small compared to its depth from the camera. For 3DGS this is almost always true: individual Gaussians are tiny relative to the scene, so the first-order Taylor expansion captures the projection well over the Gaussian's effective support.

The 2D mean and 2D covariance now fully characterize the screen-space splat that we will alpha-composite onto the image (Section 1.5).

### 1.3 color of a gaussian
This is mostly about the spherical harmonics and applying the formula (which one can refer to `sh.py`). The key is to give the view direction, i,e, the angle between the camera and gaussian center **expressed in the world coordinate**. 

### 1.4 Mahalanobis distance and impact of gaussian on a frame
This is simply applying the gaussian distribution function. For a pixel at $\mathbf{p} = (u, v)$ and the 2D Gaussian splat from the previous section (with mean and covariance both subscripted "2D"), the alpha contribution is:

$$\alpha = o \cdot \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_{2D})^\top \Sigma_{2D}^{-1} (\mathbf{p} - \boldsymbol{\mu}_{2D})\right)$$

where $o$ is the Gaussian's opacity (after sigmoid). Writing the 2×2 covariance as

$$
\Sigma_{2D} = \begin{pmatrix} a & b \cr b & c \end{pmatrix}
$$

gives a clean closed form that avoids a generic matrix inverse:

$$\alpha = o \cdot \exp\left(-\frac{1}{2(ac - b^2)} \left(c \Delta u^2 - 2 b \Delta u \Delta v + a \Delta v^2\right)\right)$$

with $\Delta u = u - \mu_u$ and $\Delta v = v - \mu_v$.

### 1.5 putting all gaussian together ($\alpha$ compositing)
Some gaussians are closer (smaller depth, i.e. $z$ axis of gaussian center inside **camera coordinate**), so they will be projected first. 
If the projected gaussians are highly dense, it will block the gaussians behind it — exactly what we want physically. Formally, for $N$ depth-sorted Gaussians and per-pixel:

$$T_i = \prod_{j < i}(1 - \alpha_j), \qquad C_{\text{pixel}} = \sum_{i=1}^{N} \alpha_i T_i c_i + T_{N+1} c_{\text{bg}}$$

$T_i$ is the **transmittance** reaching Gaussian $i$ — the fraction of light not yet blocked by Gaussians in front of it. Each Gaussian contributes its color $c_i$ weighted by both its own alpha and the transmittance that survived to reach it.
In the end, if we didn't use up all the transmittance of a pixel (i.e. gaussians impacting this pixel have not been dense enough), we will fill it using the background color. This is similar to real life — we will see the background if the objects in between are relatively transparent.

## 2. Adaptive control

This is a part that seems less intimidating compared to rendering, but can be quite intricate. The paper didn't cover a lot of details about this part, and the reference repo also has some places that don't quite align with paper.

### 2.0 scene extent

This is a quite important concept in adaptive control, as it is used (with a percentage) to determine if a guassian is large or small. In the context of reference repo, the frames at different angles are collected with cameras rotating around an axis and an isocenter. Scene extent is then computed as the **max** distance from the centroid of camera positions (with a $1.1\times$ safety margin). The choice of max rather than mean matters: it bounds the radius of the scene, not the typical camera offset. See `compute_scene_extent` in `dataset.py`.

### 2.1 what gaussians need to be work on

As described by the paper:
> Our adaptive control of the Gaussians needs to populate empty areas. It focuses on regions with missing geometric features (“underreconstruction”), but also in regions where Gaussians cover large areas in the scene (which often correspond to “over-reconstruction”). We observe that both have large view-space positional gradients. Intuitively, this is likely because they correspond to regions that are not yet well reconstructed, and the optimization tries to move the Gaussians to correct this.

Here the "view-space positional gradients" actually refers to the gradient of gaussian's 2D means (projected). The two situations can both be detected with the gradients, and we differentiate them by comparing the size of the gaussian with a percentage ($0.01$ in reference repo ) of scene extent.
### 2.2 clone small gaussians

Note that while paper mentions "creating a copy of the same size, and moving it in the direction of the positional gradient", the reference repo **didn't move the copy**.

### 2.3 split large gaussians

In this case, the original gaussian is deleted, and two smaller gaussians are created. Their centers are sampled from the parent's own 3D distribution:

$$\mathbf{x}_{\text{child}} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{parent}}, \Sigma_{\text{parent}})$$

so they remain in the same region but spread out a bit. Their scale is divided by $\phi = 1.6$, which in log-space becomes:

$$\log s_{\text{child}} = \log s_{\text{parent}} - \log 1.6$$

Rotation, opacity, and SH coefficients are inherited from the parent unchanged.

### 2.4 pruning

The paper slightly touches pruning but didn't provide many details. Based on reference repo, the step is done after cloning/splitting, and simply removes the remaining ones with small opacity **or** very large size. Note we do not remove any gaussians that were just added.

### 2.5 implementation

The implementation of adaptive control is not trivial, and one needs to pay attention to maintain the optimizer history. Adam holds per-parameter state (`exp_avg`, `exp_avg_sq`, `step`) that must stay aligned with the parameter tensors. When new Gaussians are added by concatenating to `gaussians.mean` (and the other param tensors), the optimizer state has to be updated in lockstep:

1. Pop the old parameter's state from `optimizer.state`.
2. Concatenate **zeros** to `exp_avg` and `exp_avg_sq` for the new entries (they have no gradient history yet).
3. Register the new (concatenated) tensor as a `nn.Parameter` back into `optimizer.state` and `optimizer.param_groups[i]['params']`.
4. Reattach it to the module (`setattr(gaussians, name, new_param)`) so subsequent forward passes see the right tensor.

The same dance — but with boolean masking instead of concatenation — is needed for pruning. Skipping any of these steps silently corrupts optimization: stale state entries either move the wrong Gaussians or get zeroed when they shouldn't.

## 3. Other notes

There are other places that I had considered straight-forward but found it to be worth mentioning as I compared my implementation to the reference repo.
### 3.1 different learnable parameters have different learning rates

One can find the learning rate details for each parameter group inside `main.py`

### 3.2 The opacity reset does not always happen 

Indeed this only happens **during the densification phase** (`step < densify_until_iter`). After densification ends, you've committed to your final Gaussian population and want the optimizer to refine opacities rather than reset them. The reference also adds a special trigger for **white-background scenes** (which I didn't implement): an additional opacity reset at `densify_from_iter`, which gives the initially-gray Gaussians a fresh shot at finding good opacities before serious densification starts.

### 3.3 EWA splatting smoothing

This may be more tied to the original EWA splatting paper. Simply put, we add a small constant to the diagonal of the 2D covariance:

$$\Sigma_{2D}^{\text{filtered}} = \Sigma_{2D} + h \cdot I_2 \qquad \text{with } h = 0.3$$

This guarantees every Gaussian's screen-space footprint is at least about one pixel wide. Without it, a Gaussian that projects to sub-pixel size has nowhere on the pixel grid where its alpha exceeds the contribution threshold ($1/255$), so it contributes nothing to any pixel and receives zero gradient — it gets stuck. The filter is mathematically the EWA low-pass / dilation kernel from Zwicker et al. (2002), and conceptually identical to mipmapping in classical graphics: a Nyquist-limit guard against undersampling.

And this comes with potential opacity amplification with the dilation (larger region but same opacity), so the reference repo has an antialiasing setup that rescales opacity to preserve the Gaussian's total integrated alpha. The compensation factor is the square root of the determinant ratio:

$$o_{\text{effective}} = o \cdot \sqrt{\frac{\det(\Sigma_{2D})}{\det(\Sigma_{2D}^{\text{filtered}})}}$$

When the original Gaussian is much larger than the filter, the ratio is $\approx 1$ and opacity is unchanged. When the original is sub-pixel, the ratio is small and opacity is heavily attenuated — so the dilated splat is faint, which is exactly what an aliased reconstruction should look like.

### 3.4 numerical stability considerations

This is indeed well described in the paper's Appendix C. Three things to clamp on the rendering side:

1. **Alpha is clamped at $0.99$** (max). A Gaussian with $\alpha = 1$ would zero out all transmittance behind it, including its own gradient flow. The $0.99$ cap keeps gradients alive while still giving near-opaque coverage.
2. **Negligible alpha is filtered out** ($\alpha < 1/255$ → set to $0$). Below the 8-bit color quantization threshold a contribution is unobservable, so dropping it saves compute without affecting the final image.
3. **The Mahalanobis exponent is clamped at $\le 0$** (max). Floating-point error in the inverse covariance can occasionally produce a slightly negative quadratic form, which would give $\exp(\text{positive}) > 1$ — alpha greater than the opacity, breaking the $[0,1]$ invariant.

There is also a **near-plane** cutoff in `transform_to_2dframe`: Gaussians with $z_c < 0.2$ are excluded, and $z$ is clamped before being fed into the Jacobian. Without this, Gaussians that drift behind the camera produce $1/z_c$ terms that blow up and generate NaNs that poison gradients silently.

### 3.5 Position learning rate is scaled by scene extent

The paper gives a position learning rate of $1.6 \times 10^{-4}$, but the reference actually uses `position_lr_init × spatial_lr_scale`, where `spatial_lr_scale` is the scene extent (around $4$ for `lego`). 

### 3.6 White-background compositing

NeRF-synthetic stores images as RGBA with a transparent background. Calling `Image.convert("RGB")` in `PIL`would composite against PIL's default black, but the 3DGS convention for this dataset is to composite against **white**:

```python
rgb = rgb * alpha + (1.0 - alpha)   # composite over white
```

This has to be done identically at both training and evaluation time, and the renderer's background color must match (`bg_color = [1, 1, 1]`). 

## 4. Open questions and plans

My current runs in a medium (400x400, 2x downsample) and mini (200x200, 4x downsample) scenes both lost a lot of gaussians. I ended up with ~6000 gaussians (medium scene) and ~2000 gaussians (mini scene). It is not clear if this will change with the full setup (original resolution and paper settings). I plan to try `gsplat` (with tile based rasterization) with different setups and understand the expected outcomes.