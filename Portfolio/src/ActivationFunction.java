// 活性化関数のインターフェース (隠れ層用)
public interface ActivationFunction {
    double activate(double x);
    double derivative(double x);
}