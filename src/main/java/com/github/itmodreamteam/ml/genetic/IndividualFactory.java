package com.github.itmodreamteam.ml.genetic;

public interface IndividualFactory<T extends Individual> {
    T create(double[] genes);
}
