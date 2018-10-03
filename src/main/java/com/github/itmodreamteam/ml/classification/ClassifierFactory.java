package com.github.itmodreamteam.ml.classification;

import com.github.itmodreamteam.ml.utils.collections.IntList;
import com.github.itmodreamteam.ml.utils.matrixes.Matrix;

public interface ClassifierFactory {
    Classifier build(Matrix trainFeatures, IntList trainClasses);
}
