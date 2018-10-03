package com.github.itmodreamteam.ml.classification.knn;

import com.github.itmodreamteam.ml.labs.lab1.knn.MinkowskiKnnDistMeter;
import com.github.itmodreamteam.ml.utils.matrixes.Vectors;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class MinkowskiKnnDistMeterTest {
    @Test
    public void euclidian() {
        MinkowskiKnnDistMeter euclidian = new MinkowskiKnnDistMeter(2);
        assertEquals(Math.sqrt(2), euclidian.dist(Vectors.dense(0, 0), Vectors.dense(1, 1)), 0.0001);
    }
}