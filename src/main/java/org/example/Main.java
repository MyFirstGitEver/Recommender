package org.example;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

class Vector{
    private float[] points;
    Vector(float... points){
        this.points = points;
    }

    Vector(int size){
        points = new float[size];
    }

    public float x(int i){
        return points[i];
    }

    public void setX(int pos, float value){
        points[pos] = value;
    }

    public int size(){
        return points.length;
    }

    public float distanceFrom(Vector x){
        if(size() != x.size()){
            return Float.NaN;
        }

        float total = 0;
        for(int i=0;i<x.size();i++){
            total += (x.x(i) - x(i)) * (x.x(i) - x(i));
        }

        return (float) Math.sqrt(total);
    }

    public void add(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] += v.x(i);
        }
    }

    public Vector scaleBy(float x){
        for(int i=0;i<points.length;i++){
            points[i] *= x;
        }

        return this;
    }

    public float dot(Vector w){
        if(points.length != w.size()){
            return Float.NaN;
        }

        int n = points.length;
        float total = 0;
        for(int i=0;i<n;i++){
            total += points[i] * w.x(i);
        }

        return total;
    }

    public float square(){
        float total = 0;

        for(float point : points){
            total += point * point;
        }

        return total;
    }

    public void reshape(int dimension) {
        if(dimension > points.length){
            return;
        }

        float[] p = new float[dimension];

        System.arraycopy(points, 0, p, 0, p.length);

        points = p;
    }

    public void subtract(Vector v){
        for(int i=0;i<points.length;i++){
            points[i] -= v.x(i);
        }
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        for(float point : points){
            builder.append(point);
            builder.append(" ");
        }

        return builder.toString();
    }
}

//Recommender recommender = new Recommender(y, r, 10, 500, 1.0f);
class Recommender {
    private Vector[] w, x, y, r;
    private final Vector b;
    private final float lambda;
    private final long startTime = new Date().getTime();
    private final int iterations;
    Recommender(Vector[] y, Vector[] r, int numberOfFeatures, int iterations, float lambda) {
        w = new Vector[y[0].size()];
        x = new Vector[y.length];
        b = new Vector(y[0].size());

        for(int i=0;i<y[0].size();i++){
            w[i] = new Vector(numberOfFeatures);
        }

        for(int i=0;i<y.length;i++){
            x[i] = new Vector(numberOfFeatures);
            x[i].setX(0, 1);
        }

        this.r = r;
        this.y = y;
        this.lambda = lambda;
        this.iterations = iterations;
    }

    public float cost() {
        float total = 0;

        for(int i=0;i<w.length;i++){
            for(int j=0;j<x.length;j++){
                float rated = r[j].x(i);
                double term = (double) w[i].dot(x[j]) + b.x(i) - y[j].x(i);

                total += rated * (term * term);
            }
        }

        float wTerm = 0, xTerm = 0;

        for (Vector vector : w) {
            wTerm += vector.square();
        }

        for (Vector vector : x) {
            xTerm += vector.square();
        }

        return total / 2 + (lambda / 2) * wTerm + (lambda / 2) * xTerm;
    }

    public void reset(){
        new File("w.param").delete();
        new File("b.param").delete();
        new File("x.param").delete();
    }

    public void train() throws IOException {
        File f = new File("w.param");

        if(f.exists()) {
            w = Main.loadData(f.getName(), 10);
            x = Main.loadData("x.param", 10);

            BufferedReader reader = new BufferedReader(new FileReader("b.param"));

            String line;
            int index = 0;
            while((line = reader.readLine()) != null){
                b.setX(index, Float.parseFloat(line));
                index++;
            }
        }

        int index = 0;

        Vector[] wDirections = new Vector[w.length];
        Vector bDirection = new Vector(b.size());
        Vector[] xDirections = new Vector[x.length];

        while(Math.abs(cost()) >= 0.0001 && index < iterations){
            // figure out the derivatives
            for(int i=0;i<w.length;i++){
                wDirections[i] = derivativeByW(w[i], b.x(i), x, i);
                bDirection.setX(i, derivativeByB(w[i], b.x(i), x, i));
            }

            for(int i=0;i<x.length;i++){
                xDirections[i] = derivativeByX(x[i], b, w, i);
            }
            //CODE END HERE ###

            for(int i=0;i<w.length;i++){
                w[i].subtract(wDirections[i].scaleBy(6e-4f));
            }

            for(int i=0;i<x.length;i++){
                x[i].subtract(xDirections[i].scaleBy(6e-4f));
            }

            b.subtract(bDirection.scaleBy(6e-4f));

            index++;
        }

        From.from(startTime);
        saveParams();
    }

    private void saveParams() throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter("w.param"));
        for(Vector v : w){
            writer.write(v.toString());
            writer.newLine();
        }

        writer.close();

        writer = new BufferedWriter(new FileWriter("x.param"));
        for(Vector v : x){
            writer.write(v.toString());
            writer.newLine();
        }

        writer.close();

        writer = new BufferedWriter(new FileWriter("b.param"));
        for(int i=0;i<b.size();i++){
            writer.write(Float.toString(b.x(i)));
            writer.newLine();
        }

        writer.close();
    }
    private float predict(Vector x, Vector w, float b) {
        return x.dot(w) + b;
    }

    private Vector derivativeByW(Vector w, float b, Vector[] x, int userNum) {
        Vector derivative = new Vector(w.size());

        int features = w.size();

        for(int i=0;i<features;i++) {
            for (int j=0;j<x.length;j++) {
                float curr = derivative.x(i);

                curr += r[j].x(userNum) * x[j].x(i) *
                        (predict(x[j], w, b) - y[j].x(userNum));
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) + lambda * w.x(i));
        }

        return derivative;
    }

    private Vector derivativeByX(Vector w, Vector b, Vector[] x, int itemNum) {
        Vector derivative = new Vector(w.size());

        int features = w.size();

        for(int i=0;i<features;i++){
            for (int j=0;j<x.length;j++) {
                float curr = derivative.x(i);

                curr += r[itemNum].x(j) * x[j].x(i) *
                        (predict(w, x[j], b.x(j)) - y[itemNum].x(j));
                derivative.setX(i, curr);
            }

            derivative.setX(i, derivative.x(i) + lambda * w.x(i));
        }

        return derivative;
    }

    private float derivativeByB(Vector w, float b, Vector[] x, int userNum) {
        float total = 0;

        for (int i=0;i<x.length;i++) {
            total += r[i].x(userNum) * (predict(x[i], w, b) - y[i].x(userNum));
        }

        return total;
    }
}

class From{
    static void from(long startTime){
        long doneTime = (new Date().getTime() - startTime) / 1000;
        long hour = 0, min = 0, sec;

        if(doneTime < 60){
            sec = doneTime;
        }
        else if(doneTime < 3600){
            min = doneTime / 60;
            sec = doneTime % 60;
        }
        else{
            hour = doneTime / 3600;
            min = doneTime % 3600;
            sec = min % 60;
            min /= 60;
        }

        System.out.println(hour + ":" + min + ":" + sec);
    }
}

public class Main {
    public static void main(String[] args) throws IOException {
        Vector[] y = loadData("y.txt", 444);
        Vector[] r = loadData("r.txt", 444);

        Recommender recommender = new Recommender(y, r, 10, 100, 1.0f);
        //recommender.reset();
        recommender.train();

        System.out.println(recommender.cost());
    }

    static Vector[] loadData(String path, int vectorSize) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(path));

        String x;
        List<Vector> list = new ArrayList<>();

        int lastLength = 0;
        Vector v = null;
        while((x = reader.readLine()) != null){
            String data = x.trim().replaceAll(" +", " ");
            String[] entries = data.split(" ");

            if(lastLength == 0){
                v = new Vector(vectorSize);
            }

            for(int i=0;i<entries.length;i++){
                v.setX(i + lastLength, Float.parseFloat(entries[i]));
            }

            if(lastLength + entries.length == vectorSize){
                lastLength = 0;
                list.add(v);
            }
            else{
                lastLength += entries.length;
            }
        }

        reader.close();

        return list.toArray(new Vector[0]);
    }
}