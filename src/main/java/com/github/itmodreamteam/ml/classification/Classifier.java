package com.github.itmodreamteam.ml.classification;

public interface Classifier<F, A> {
    A classify(F features);
}
