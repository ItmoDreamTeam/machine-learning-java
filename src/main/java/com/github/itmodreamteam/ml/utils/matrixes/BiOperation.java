package com.github.itmodreamteam.ml.utils.matrixes;

@FunctionalInterface
public interface BiOperation {
    BiOperation DO_NOTHING = (f, s) -> f;
    BiOperation SUM = (f, s) -> f + s;
    BiOperation DIFF = (f, s) -> f - s;
    BiOperation MULT = (f, s) -> f * s;
    BiOperation DIVIDE = (f, s) -> f / s;

    double apply(double first, double second);

    default BiOperation andThen(BiOperation after, double value) {
        return (first, second) -> after.apply(apply(first, second), value);
    }
}
