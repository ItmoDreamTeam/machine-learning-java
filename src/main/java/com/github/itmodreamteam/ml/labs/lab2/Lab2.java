package com.github.itmodreamteam.ml.labs.lab2;

import com.github.itmodreamteam.ml.regression.*;
import com.github.itmodreamteam.ml.utils.ClassPathResources;
import com.github.itmodreamteam.ml.utils.io.Csv;
import com.github.itmodreamteam.ml.utils.matrixes.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Scanner;

public class Lab2 {
    private static final Scanner SCANNER = new Scanner(System.in);

    private static final Logger LOGGER = LoggerFactory.getLogger(Lab2.class);

    public static void main(final String... args) throws Exception {
        Csv csv = Csv.read(ClassPathResources.getFile("prices.txt"), ",", false);
        Matrix features = Matrixes.dense(csv.doubles("area", "rooms"));
        Vector prices = Vectors.dense(csv.doubles("price"));
        LinearRegressionFactory factory1 = new GradientDescentLinearRegressionFactory(50000, 0.001);
        LinearRegressionFactory factory2 = new NormalEquationSolverLinearRegressionFactory();
        LinearRegressionFactory genetic1 = new GeneticLinearRegressionFactory(
                100,
                80,
                20,
                false,
                20
        );
        LinearRegression regression1 = factory1.make(features, prices);
        LinearRegression regression2 = factory2.make(features, prices);
        LinearRegression regression3 = genetic1.make(features, prices);

        LOGGER.info("{}", regression1);
        LOGGER.info("{}", regression2);
        LOGGER.info("{}", regression3);
        test(regression1);
        test(regression2);
        test(regression3);
    }

    private static void test(LinearRegression regression) {
        while (true) {
            String line = SCANNER.nextLine();
            if (line.equalsIgnoreCase("quit")) {
                break;
            } else {
                Vector features = parseDoubles(line);
                System.out.println(regression.answer(features));
            }
        }
    }

    private static Vector parseDoubles(String line) {
        double[] doubles = Arrays.stream(line.split(","))
                .map(String::trim)
                .mapToDouble(Double::parseDouble)
                .toArray();
        return Vectors.dense(doubles);
    }
}
