package com.github.itmodreamteam.ml.utils.matrixes;

@FunctionalInterface
public interface EnrichFunction {
    EnrichFunction QUADRATIC = new EnrichFunction() {
        @Override
        public double[] enrich(double[] source) {
            double[] target = new double[source.length];
            for (int i = 0; i < source.length; ++i) {
                target[i] = source[i] * source[i];
            }
            return target;
        }
    };

    EnrichFunction MULT_ALL_PAIRS = new EnrichFunction() {
        @Override
        public double[] enrich(double[] source) {
            double[] target = new double[source.length * source.length];
            int index = 0;
            for (double aSource : source) {
                for (double aSource1 : source) {
                    target[index++] = aSource * aSource1;
                }
            }
            return target;
        }
    };

    double[] enrich(double[] source);
}
