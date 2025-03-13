// Softmax 活性化関数 (出力層用、分類問題向け)
class Softmax implements OutputActivationFunction {
    @Override
    public double[] activate(double[] x) {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (double val : x) {
            maxVal = Math.max(maxVal, val);
        }
        double expSum = 0;
        double[] expValues = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            expValues[i] = Math.exp(x[i] - maxVal); // オーバーフロー対策
            expSum += expValues[i];
        }
        double[] softmaxOutput = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            softmaxOutput[i] = expValues[i] / expSum;
        }
        return softmaxOutput;
    }

    @Override
    public double[] derivative(double[] x, double[] output) {
        // ソフトマックス関数の微分は出力自体を使って計算できる (今回は交差エントロピー誤差関数と組み合わせるため、微分は不要)
        // 誤差逆伝播の計算内で、交差エントロピー誤差関数の微分とソフトマックス関数の微分を組み合わせた形になる
        return output; // 便宜的に出力をそのまま返す (実際には使用されない)
    }
}