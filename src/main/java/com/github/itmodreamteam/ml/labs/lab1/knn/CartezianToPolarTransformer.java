package com.github.itmodreamteam.ml.labs.lab1.knn;

import com.github.itmodreamteam.ml.utils.matrixes.*;

public class CartezianToPolarTransformer implements DimensionTransformer {
    private final MatrixFactory factory;

    public CartezianToPolarTransformer() {
        factory = Matrixes.factory();
    }

    public CartezianToPolarTransformer(MatrixFactory factory) {
        this.factory = factory;
    }

    @Override
    public Matrix transform(Matrix source) {
        Vector x = source.col(0);
        Vector y = source.col(1);
        Vector r = x.power(2).plus(y.power(2)).power(0.5);
        Vector theta = y.devide(x).assign(Operation.ATAN);
        return factory.joinColumns(r, theta);
    }
}
