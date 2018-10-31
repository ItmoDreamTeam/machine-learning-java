package com.github.itmodreamteam.ml.genetic;

import java.util.Arrays;

public class Individual {
    private final double[] genes;

    public Individual(double[] genes) {
        this.genes = genes;
    }

    public double[] getGenes() {
        return genes;
    }

    @Override
    public String toString() {
        return Arrays.toString(genes);
    }
}
