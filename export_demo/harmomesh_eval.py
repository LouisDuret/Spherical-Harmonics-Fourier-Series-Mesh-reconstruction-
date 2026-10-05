"""Évaluateur HarmoMesh (généré automatiquement). Usage : python harmomesh_eval.py coeffs.json"""
import json, sys
import numpy as np


def real_sh(L, theta, phi):
    """Harmoniques réelles orthonormées, phase de Condon-Shortley ; colonne l*l + l + m."""
    ct, st = np.cos(theta), np.sin(theta)
    Y = np.zeros((len(theta), (L + 1) ** 2))
    pmm = np.full_like(ct, 0.28209479177387814)
    for m in range(L + 1):
        if m:
            pmm = -np.sqrt((2 * m + 1) / (2 * m)) * st * pmm
        p1, p2 = pmm, 0 * ct
        for l in range(m, L + 1):
            if l > m:
                a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
                b = np.sqrt(max(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1), 0))
                p1, p2 = a * (ct * p1 - b * p2), p1
            i = l * l + l
            if m == 0:
                Y[:, i] = p1
            else:
                Y[:, i + m] = np.sqrt(2) * p1 * np.cos(m * phi)
                Y[:, i - m] = np.sqrt(2) * p1 * np.sin(m * phi)
    return Y


def evaluate(d, theta=None, phi=None, xyz=None):
    """A : rayon r(theta, phi) ; B : points (x, y, z)(theta, phi) ; C : f(x, y, z), surface f = 0.
    Entrées et sorties dans le repère d'origine du mesh."""
    c = np.array(d["coefficients"], float)
    s, o = d["normalization"]["scale"], np.array(d["normalization"]["center"])
    if d["method"] == "C":
        K, h = d["kmax"], d["half_size"]
        u = ((np.atleast_2d(xyz) - o) / s + h) / (2 * h)
        k = np.stack(np.meshgrid(*[np.arange(-K, K + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        return (np.exp(2j * np.pi * u @ k.T) @ (c[:, 0] + 1j * c[:, 1])).real * s
    Y = real_sh(d["lmax"], np.atleast_1d(np.asarray(theta, float)), np.atleast_1d(np.asarray(phi, float)))
    return (Y @ c) * s + (o if d["method"] == "B" else 0)


if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print("méthode", d["method"], "- degré", d["lmax"], "-", len(d["coefficients"]), "coefficients")
    if d["method"] == "C":
        print("f(centre) =", evaluate(d, xyz=[d["normalization"]["center"]]))
    else:
        print("valeur en (theta, phi) = (pi/2, 0) :", evaluate(d, [np.pi / 2], [0.0]))
