package com.github.itmodreamteam.ml.utils.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.*;

public class Csv {
    private final Map<String, List<String>> content;
    private final int numberOfRecords;

    private Csv(Map<String, List<String>> content, int numberOfRecords) {
        this.content = content;
        this.numberOfRecords = numberOfRecords;
    }

    public static Csv read(Reader reader, String separator, boolean ordered) throws IOException {
        BufferedReader br = new BufferedReader(reader);
        String header = br.readLine();
        String[] fields = header.split(separator);
        Map<String, List<String>> content = new HashMap<>();
        Map<Integer, String> index = new HashMap<>();
        for (int fieldNumber = 0; fieldNumber < fields.length; ++fieldNumber) {
            String field = fields[fieldNumber];
            content.put(field.trim(), new ArrayList<>());
            index.put(fieldNumber, field.trim());
        }
        List<String> lines = new ArrayList<>();
        {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        }
        br.close();
        if (!ordered) {
            Collections.shuffle(lines);
        }
        for (String line : lines) {
            String[] values = line.split(separator);
            for (int fieldNumber = 0; fieldNumber < fields.length; ++fieldNumber) {
                content.get(index.get(fieldNumber)).add(values[fieldNumber].trim());
            }
        }
        return new Csv(content, lines.size());
    }

    public int records() {
        return numberOfRecords;
    }

    public double[] doubles(String field) {
        return content.get(field).stream()
                .mapToDouble(Double::parseDouble)
                .toArray();
    }

    public double[][] doubles(String... fields) {
        double[][] result = new double[numberOfRecords][fields.length];
        for (int fieldNumber = 0; fieldNumber < fields.length; ++fieldNumber) {
            String field = fields[fieldNumber];
            double[] column = doubles(field);
            for (int i = 0; i < numberOfRecords; ++i) {
                result[i][fieldNumber] = column[i];
            }
        }
        return result;
    }

    public int[] ints(String field) {
        return content.get(field).stream()
                .mapToInt(Integer::parseInt)
                .toArray();
    }

    public int[][] ints(String... fields) {
        int[][] result = new int[numberOfRecords][fields.length];
        for (int fieldNumber = 0; fieldNumber < fields.length; ++fieldNumber) {
            String field = fields[fieldNumber];
            int[] column = ints(field);
            for (int i = 0; i < numberOfRecords; ++i) {
                result[i][fieldNumber] = column[i];
            }
        }
        return result;
    }
}
