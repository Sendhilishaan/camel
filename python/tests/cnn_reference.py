"""Small float64 Python references for CNN checks; no runtime backend or numpy.

The references scatter backward contributions while CUDA gathers them, so an
indexing mistake in one implementation is less likely to be repeated in both.
"""
from camel import array as ca


def conv2d(x, w, b=None, stride=(1, 1), padding=(0, 0), grad=None):
    n, channels, h, width = x.shape
    filters, _, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    oh, ow = (h + 2*ph - kh) // sh + 1, (width + 2*pw - kw) // sw + 1
    out = ca.zeros((n, filters, oh, ow))
    dx, dw, db = ca.zeros_like(x), ca.zeros_like(w), ca.zeros((1, filters))
    for batch in range(n):
        for f in range(filters):
            for oy in range(oh):
                for ox in range(ow):
                    oi = (batch, f, oy, ox)
                    value = b[0, f] if b is not None else 0.0
                    g = grad[oi] if grad is not None else 0.0
                    db[0, f] += g
                    for channel in range(channels):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy, ix = oy*sh + ky - ph, ox*sw + kx - pw
                                if 0 <= iy < h and 0 <= ix < width:
                                    xi, wi = (batch, channel, iy, ix), (f, channel, ky, kx)
                                    value += x[xi] * w[wi]
                                    dx[xi] += g * w[wi]
                                    dw[wi] += g * x[xi]
                    out[oi] = value
    return out, dx, dw, db


def maxpool2d(x, kernel, stride, grad=None):
    n, channels, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    oh, ow = (h-kh) // sh + 1, (w-kw) // sw + 1
    out, dx, indices = ca.zeros((n, channels, oh, ow)), ca.zeros_like(x), []
    for batch in range(n):
        for channel in range(channels):
            for oy in range(oh):
                for ox in range(ow):
                    window = [(batch, channel, oy*sh+ky, ox*sw+kx)
                              for ky in range(kh) for kx in range(kw)]
                    winner = max(window, key=lambda idx: x[idx]) # first maximum
                    out[batch, channel, oy, ox] = x[winner]
                    indices.append(((batch*channels + channel)*h + winner[2])*w + winner[3])
                    if grad is not None:
                        dx[winner] += grad[batch, channel, oy, ox]
    return out, dx, indices
