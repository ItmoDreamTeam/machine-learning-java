package com.github.itmodreamteam.ml.classification.knn;

import com.github.itmodreamteam.ml.utils.matrixes.Matrix;

public interface DimensionTransformer {
    DimensionTransformer CARTESIAN_TO_POLAR = new CartezianToPolarTransformer();

    DimensionTransformer IDENTITY = source -> source;

    Matrix transform(Matrix source);
}
