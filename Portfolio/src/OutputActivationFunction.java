// 出力層の活性化関数のインターフェース (Softmaxなど)
public interface OutputActivationFunction {
    double[] activate(double[] x); // 入力はdouble配列
    double[] derivative(double[] x, double[] output);
}