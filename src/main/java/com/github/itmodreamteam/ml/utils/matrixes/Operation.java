package com.github.itmodreamteam.ml.utils.matrixes;

@FunctionalInterface
public interface Operation {
    Operation IDENTITY = source -> source;

    Operation SQUARE = source -> source * source;

    Operation INVERSE = source -> 1.0 / source;

    Operation ABS = source -> source < 0 ? -source : source;

    double apply(double source);

    default Operation andThen(Operation after) {
        return source -> after.apply(apply(source));
    }
}
