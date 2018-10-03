package com.github.itmodreamteam.ml.utils.matrixes;

public interface AggregateFunction {
    AggregateFunction SUM = (accumulator, value) -> accumulator + value;

    double reduce(double accumulator, double value);
}
