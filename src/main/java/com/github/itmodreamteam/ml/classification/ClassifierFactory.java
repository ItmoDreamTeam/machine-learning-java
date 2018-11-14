package com.github.itmodreamteam.ml.classification;

import com.github.itmodreamteam.ml.validation.Samples;

public interface ClassifierFactory<F, A> {
    Classifier<F, A> build(Samples<F, A> train);
}
