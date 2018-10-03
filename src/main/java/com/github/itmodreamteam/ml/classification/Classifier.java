package com.github.itmodreamteam.ml.classification;

import com.github.itmodreamteam.ml.utils.matrixes.Vector;

public interface Classifier {
    int classify(Vector features);
}
